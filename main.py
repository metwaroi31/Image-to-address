
# from groundingdino.util.inference import (
#     load_model,
#     load_image,
#     predict,
#     crop_image,
#     annotate
# )
# from image_utils.read_metadata import (
#     write_metadata_to_csv
# )
from openAI_service.LLMs import GPTLLM
# from vintern_llava.LLMs import ImageOCRLLM
import glob
# from groundingdino.detector import GroundingDinoDetector
import csv
import pandas as pd

# model = load_model(
#     "groundingdino/config/GroundingDINO_SwinT_OGC.py",
#     "weights/groundingdino_swint_ogc.pth"
# )

# OCR_LLM_MODEL = ImageOCRLLM()
GPT_LLM = GPTLLM()

# DETECT_MODEL = GroundingDinoDetector(model=model)

while True:
    file_path = '/content/GSM_Image-to-address/data.csv'  # Replace with your file's path
    df = pd.read_csv(file_path) 
    df = df.head(50)

    # # Temporarily set pandas to display the full string without truncation
    # pd.set_option('display.max_colwidth', None)

    # # Print all values in the 'ocr_result' column
    # print(df['ocr_result'])

    # # Reset to default (optional)
    # pd.reset_option('display.max_colwidth')
    # break

    keys = ["shop_name","address","phone_number","email","category","product","district","street_no","street_name","city","ward","ocr_result","file_name"]
    with open("report.csv", mode='a', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=keys)
        csv_file.seek(0, 2)  # Move to end of file to check if empty
        if csv_file.tell() == 0:  # If file is empty, write header
            writer.writeheader()

    # Check if the column "ocr_results" exists
        if "ocr_result" in df.columns:
            # Iterate over the rows in "ocr_results" column
            for ocr_values in df["ocr_result"]:
                json_of_point = GPT_LLM.get_poi_from_text(ocr_values)
                json_of_point["ocr_result"] = ocr_values
                filtered_point = {key: json_of_point.get(key, None) for key in keys}
                writer.writerow(filtered_point)
        else:
            print("Column 'ocr_result' not found in the file.")

          
    break
