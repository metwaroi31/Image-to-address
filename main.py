
from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    crop_image,
    annotate
)
from image_utils.read_metadata import (
    write_metadata_to_csv
)
from openAI_service.LLMs import GPTLLM
from vintern_llava.LLMs import ImageOCRLLM
import glob
from groundingdino.detector import GroundingDinoDetector
import csv

# model = load_model(
#     "groundingdino/config/GroundingDINO_SwinT_OGC.py",
#     "weights/groundingdino_swint_ogc.pth"
# )

OCR_LLM_MODEL = ImageOCRLLM()
GPT_LLM = GPTLLM()

# DETECT_MODEL = GroundingDinoDetector(model=model)

while True:
    # TODO: For each image Get the following
    # Get multiple addresses and shop names
    # Get lat/lng
    # save some information for training
    # integrate with MapAPI
    # for image_file in glob.glob("crop_images/*"):
    #     image_exif_data = get_exif_data(image_file)
    #     store_sign_images = DETECT_MODEL.predict_billboards(image_file, image_exif_data)
    #     for image_frame in store_sign_images:
    #         ocr_values = LLM_MODEL.extract_text_images(image_frame)
            
            # LLM_MODEL.send_poi_info_to_db(image_frame, image_exif_data)
    """Writes the list of metadata dictionaries to a CSV file."""
    # Extract keys for the CSV header from the first dictionary
    keys = ["shop_name","address","phone_number","email","category","product","district","street_no","street_name","city","ward","ocr_result","file_name"]
    with open("report.csv", mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        for image_file in glob.glob("crop_images/*"):
            print("Processing image:", image_file)
            try:
                ocr_values = OCR_LLM_MODEL.extract_text_images(image_file)
                json_of_point = {
                    "ocr_result": ocr_values,
                    "file_name": image_file,
                }
                # json_of_point = GPT_LLM.get_poi_from_text(ocr_values)
                # json_of_point["ocr_result"] = ocr_values
                # json_of_point["file_name"] = image_file
                writer.writerow(json_of_point)
            except Exception as error:
                print (str(error))
                print ("bad images")
    break
