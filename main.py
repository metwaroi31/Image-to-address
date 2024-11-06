from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    crop_image,
    annotate
)
from image_utils.read_metadata import (
    get_exif_data
)
from vintern_llava.LLMs import ImageOCRLLM
import glob
from groundingdino.detector import GroundingDinoDetector


model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

LLM_MODEL = ImageOCRLLM()
DETECT_MODEL = GroundingDinoDetector(model=model)

while True:
    # TODO: For each image Get the following
    # Get multiple addresses and shop names
    # Get lat/lng
    # save some information for training
    # integrate with MapAPI
    for image_file in glob.glob("images/*"):
        image_exif_data = get_exif_data(image_file)
        store_sign_images = DETECT_MODEL.predict_billboards(image_file, image_exif_data)
        for image_frame in store_sign_images:
            LLM_MODEL.send_poi_info_to_db(image_frame, image_exif_data)
    break
