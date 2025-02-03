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
            text_prompt = "Billboard contains logo and address .Traffic lights. Traffic signal. Taxi sign. Vacant sign. Street number sign. Prohibition sign. Warning sign. Mandatory sign.  Business signage. Awning business sign contains logo and address. Construction sign. Construction barrier. Advertisement on utility pole. Graffiti . Utility Pole. Street poster. Advertisement poster. Business poster.",
            box_threshold = 0.35,
            text_threshold  = 0.20,
        ) -> None:
        self.model = model
        self.TEXT_PROMPT = text_prompt
        self.BOX_THRESHOLD = box_threshold
        self.TEXT_THRESHOLD = text_threshold
        self.KEYWORD_DETECT_BOX = ["business billboard",
                                   "business signage", 
                                   "business sign",
                                   "awning business billboard",
                                   "awning business sign",
                                   "awning business signage",
                                   ]

    def predict_billboards(self, image_file, image_exif_data):
        input_file_names = convert_streetview_to_normal_image(image_file, image_exif_data)
        store_sign_images = []
        for input_file_name in input_file_names:
            print (input_file_name)
            start_time = time.time()
            image_source, image = load_image(
                input_file_name
            )
            boxes, _ , logits, phrases = predict(
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
                phrases,
                input_file_name,
                keywords = self.KEYWORD_DETECT_BOX
            ))
            annotate(
                image_source=image_source,
                boxes=boxes,
                logits=logits,
                phrases=phrases
            )
        return store_sign_images
