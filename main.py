from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    crop_image,
    annotate
)
import time
import glob
from image_utils.image_processor import (
    from_tensor_to_pixels,
    convert_streetview_to_normal_image
)
from image_utils.read_metadata import (
    get_exif_data,
    write_metadata_to_csv
)
model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

TEXT_PROMPT = "billboard . sign . advertisement"
BOX_TRESHOLD = 0.20
TEXT_TRESHOLD = 0.20
while True:
    # TODO: For each image Get the following
    # Get multiple addresses and shop names
    # Get lat/lng
    # save some information for training
    # integrate with MapAPI
    for image_file in glob.glob("images/*"):
        input_file_name = convert_streetview_to_normal_image(image_file)
        print (image_file)
        start_time = time.time()
        image_exif_data = get_exif_data(image_file)
        json_of_image = image_exif_data['File Name']

        image_source, image = load_image(
            input_file_name
        )
        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        
        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time} seconds")
        store_sign_images = crop_image(
            image_source,
            boxes,
            input_file_name
        )
        
        annotated_frame = annotate(
            image_source=image_source,
            boxes=boxes,
            logits=logits,
            phrases=phrases
        )
