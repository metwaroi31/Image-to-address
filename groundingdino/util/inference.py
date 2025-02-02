from typing import Tuple, List

import cv2
import numpy as np
import supervision as sv
import os
import torch
from PIL import Image
from torchvision.ops import box_convert
import bisect
import time
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.misc import clean_state_dict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import get_phrases_from_posmap

import json
# ----------------------------------------------------------------------------------------------------------------------
# OLD API
# ----------------------------------------------------------------------------------------------------------------------


def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(
        model_checkpoint_path,
        map_location="cpu"
        # weights_only=True
    )
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image_source = Image.open(image_path).convert("RGB")
    image = np.asarray(image_source)
    image_transformed, _ = transform(image_source, None)
    return image, image_transformed


def predict(
        model,
        image: torch.Tensor,
        caption: str,
        box_threshold: float,
        text_threshold: float,
        device: str = "cuda",
        remove_combined: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    caption = preprocess_caption(caption=caption)

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])

    prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # prediction_logits.shape = (nq, 256)
    prediction_boxes = outputs["pred_boxes"].cpu()[0]  # prediction_boxes.shape = (nq, 4)

    mask = prediction_logits.max(dim=1)[0] > box_threshold
    logits = prediction_logits[mask]  # logits.shape = (n, 256)
    boxes = prediction_boxes[mask]  # boxes.shape = (n, 4)

    if logits.shape[0] == 0:
        return torch.empty(0), torch.empty(0), torch.empty(0), []

    
    scores = logits.max(dim=1)[0]
    
    tokenizer = model.tokenizer
    tokenized = tokenizer(caption)
    
    if remove_combined:
        sep_idx = [i for i in range(len(tokenized['input_ids'])) if tokenized['input_ids'][i] in [101, 102, 1012]]
        
        phrases = []
        for logit in logits:
            max_idx = logit.argmax()
            insert_idx = bisect.bisect_left(sep_idx, max_idx)
            right_idx = sep_idx[insert_idx]
            left_idx = sep_idx[insert_idx - 1]
            phrases.append(get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer, left_idx, right_idx).replace('.', ''))
    else:
        phrases = [
            get_phrases_from_posmap(logit > text_threshold, tokenized, tokenizer).replace('.', '')
            for logit
            in logits
        ]

    return boxes, scores, logits, phrases

def crop_image(
        image_source: np.ndarray,
        boxes: torch.Tensor,
        phrases: list,
        full_image_file_name: str,
        keywords: list 
    ):
    
    BASE_FOLDER_CROP_IMAGE = "crop_images/"
    # TODO: please modify this
    image_file_name = full_image_file_name.split('/')
    image_file_name = image_file_name[len(image_file_name) - 1]
    return_frames = []
    
    if boxes.numel() == 0:  # Không có box nào
        print("No boxes detected.")
        return return_frames
    
    h, w, _ = image_source.shape
    boxes_ = boxes * torch.Tensor([w, h, w, h])
    xyxys = box_convert(boxes=boxes_, in_fmt="cxcywh", out_fmt="xyxy").numpy().tolist()

    for i, xyxy in enumerate(xyxys):
        label = phrases[i].lower()
        if not any(keyword.lower() in label for keyword in keywords):
            print(f"Skipping box with label: {label}")
            continue
        
        x1, y1, x2, y2 = [int(_) for _ in xyxy]
        width = x2 - x1
        height = y2 - y1
        ratio = width/height
        area = width * height
        # min_area = 100000
        # print(f"Processing box with area: {area}")
        # For billboards that are not 
        #TODO: 
        # if ratio > 3 and area < 250000:
        #     continue
        # if width < 800 and height < 800:
        #     continue
        # if x1 < 700:
        #     continue
        # if x1 > 3500:
        #     continue
        
        # if  area < min_area :
        #     print(f"Skipping box with area: {area}")    
        #     continue
        
        # if 0.3125 <= ratio <= 2.5 : 
        #     continue
        cropped_frame = image_source[y1:y2, x1:x2]
        return_frames.append(cropped_frame)
        timestamp = time.time()
        Image.fromarray(cropped_frame).save(
            BASE_FOLDER_CROP_IMAGE + image_file_name + "_" + str(timestamp) + ".jpg"
        )
    return return_frames

def annotate(image_source: np.ndarray, 
             boxes: torch.Tensor, 
             logits: torch.Tensor, 
             phrases: List[str]) -> np.ndarray:
    """    
    This function annotates an image with bounding boxes and labels.

    Parameters:
    image_source (np.ndarray): The source image to be annotated.
    boxes (torch.Tensor): A tensor containing bounding box coordinates.
    logits (torch.Tensor): A tensor containing confidence scores for each bounding box.
    phrases (List[str]): A list of labels for each bounding box.

    Returns:
    np.ndarray: The annotated image.
    """
    
    if boxes.numel() == 0:  # Không có box nào
        print("No boxes detected.")
        return cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    detections = sv.Detections(xyxy=xyxy)
    BASE_FOLDER_ANNOTATE_IMAGE = "images_annotated/"

    # labels = [
    #     f"{phrase} {logit:.2f}"
    #     for phrase, logit
    #     in zip(phrases, logits)
    # ]
    labels = [
            f"{phrase} {logit.item():.2f}"
            for phrase, logit
            in zip(phrases, logits.max(dim=1)[0])
    ]

    bbox_annotator = sv.BoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
    label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)
    annotated_frame = cv2.cvtColor(image_source, cv2.COLOR_RGB2BGR)
    annotated_frame = bbox_annotator.annotate(scene=annotated_frame, detections=detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
    
    timestamp = time.time()
    Image.fromarray(annotated_frame).save(
    BASE_FOLDER_ANNOTATE_IMAGE + str(timestamp) + ".jpg"
    )
    return annotated_frame

def save_predictions_to_json(image_path, 
                             boxes, 
                             scores,
                             phrases,
                             output_folder):
    
    
    """
    Lưu dự đoán bounding box và điểm số thành file JSON.
    """
    # Ensure image_path is a string
    
    file_name = os.path.basename(image_path).split('.')[0] + "_results.json"
    output_path = os.path.join(output_folder, file_name)

    # # Tạo dictionary kết quả
    # results = {
    #     "image_path": image_path,
    #     "detections": [
    #         {
    #             "box": box.tolist() if isinstance(box, torch.Tensor) else box,
    #             "score": float(score) if isinstance(score, torch.Tensor) else score,
    #             "label": phrase
    #         }
    #         for box, score, phrase in zip(boxes, scores, phrases)
    #     ]
    # }
        # Kiểm tra nếu boxes trống
    if boxes.numel() == 0:  # Không có box nào
        results = {
            "image_path": image_path,
            "detections": []  # Không có box nào
        }
    else:
        # Tạo dictionary kết quả nếu có boxes
        results = {
            "image_path": image_path,
            "detections": [
                {
                    "box": box.tolist() if isinstance(box, torch.Tensor) else box,
                    "score": float(score) if isinstance(score, torch.Tensor) else score,
                    "label": phrase
                }
                for box, score, phrase in zip(boxes, scores, phrases)
            ]
        }

    # Ghi kết quả ra file JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    # BASE_FOLDER_SAVE_PREDICTIONS_TO_JSON = "predictions_json/"
    print(f"Results saved to {output_folder}")


# ----------------------------------------------------------------------------------------------------------------------
# NEW API
# ----------------------------------------------------------------------------------------------------------------------


class Model:

    def __init__(
        self,
        model_config_path: str,
        model_checkpoint_path: str,
        device: str = "cuda"
    ):
        self.model = load_model(
            model_config_path=model_config_path,
            model_checkpoint_path=model_checkpoint_path,
            device=device
        ).to(device)
        self.device = device

    def predict_with_caption(
        self,
        image: np.ndarray,
        caption: str,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ) -> Tuple[sv.Detections, List[str]]:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections, labels = model.predict_with_caption(
            image=image,
            caption=caption,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections, labels=labels)
        """
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold, 
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        return detections, phrases

    def predict_with_classes(
        self,
        image: np.ndarray,
        classes: List[str],
        box_threshold: float,
        text_threshold: float
    ) -> sv.Detections:
        """
        import cv2

        image = cv2.imread(IMAGE_PATH)

        model = Model(model_config_path=CONFIG_PATH, model_checkpoint_path=WEIGHTS_PATH)
        detections = model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )


        import supervision as sv

        box_annotator = sv.BoxAnnotator()
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        """
        caption = ". ".join(classes)
        processed_image = Model.preprocess_image(image_bgr=image).to(self.device)
        boxes, logits, phrases = predict(
            model=self.model,
            image=processed_image,
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device)
        source_h, source_w, _ = image.shape
        detections = Model.post_process_result(
            source_h=source_h,
            source_w=source_w,
            boxes=boxes,
            logits=logits)
        class_id = Model.phrases2classes(phrases=phrases, classes=classes)
        detections.class_id = class_id
        return detections

    @staticmethod
    def preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_pillow = Image.fromarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        image_transformed, _ = transform(image_pillow, None)
        return image_transformed

    @staticmethod
    def post_process_result(
            source_h: int,
            source_w: int,
            boxes: torch.Tensor,
            logits: torch.Tensor
    ) -> sv.Detections:
        boxes = boxes * torch.Tensor([source_w, source_h, source_w, source_h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        confidence = logits.numpy()
        return sv.Detections(xyxy=xyxy, confidence=confidence)

    @staticmethod
    def phrases2classes(phrases: List[str], classes: List[str]) -> np.ndarray:
        class_ids = []
        for phrase in phrases:
            for class_ in classes:
                if class_ in phrase:
                    class_ids.append(classes.index(class_))
                    break
            else:
                class_ids.append(None)
        return np.array(class_ids)
