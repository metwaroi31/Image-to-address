from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    crop_image,
    annotate
)
from image_utils.image_processor import (
    convert_streetview_to_normal_image
)
import time


# This class is to handle configuration
# And inference to GroundingDino
class GroundingDinoDetector:
        
    def __init__(
            self,
            model,
            text_prompt = "billboard . sign . advertisement",
            box_threshold = 0.20,
            text_threshold  = 0.20,
        ) -> None:
        self.model = model
        self.TEXT_PROMPT = text_prompt
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold

    def predict_billboards(self, image_file, image_exif_data):
        input_file_names = convert_streetview_to_normal_image(image_file, image_exif_data)
        store_sign_images = []
        for input_file_name in input_file_names:
            print (input_file_name)
            start_time = time.time()
            image_source, image = load_image(
                input_file_name
            )
            boxes, logits, phrases = predict(
                model=self.model,
                image=image,
                caption=self.TEXT_PROMPT,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_THRESHOLD
            )
            
            end_time = time.time()
            print(f"Elapsed time: {end_time - start_time} seconds")
            store_sign_images.extend(crop_image(
                image_source,
                boxes,
                input_file_name
            ))
            annotate(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases
            )
        return store_sign_images
