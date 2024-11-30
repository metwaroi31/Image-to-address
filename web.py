import os
from flask import Flask, request, jsonify
from groundingdino.util.inference import (
    load_model
)
from image_utils.read_metadata import (
    get_exif_data
)
from apis.map import send_post_request
from openAI_service.LLMs import GPTLLM
from vintern_llava.LLMs import ImageOCRLLM
import json
import glob
from groundingdino.detector import GroundingDinoDetector
import asyncio

app = Flask(__name__)
UPLOAD_FOLDER = "D:\\content_detection\\source_code\\Image-to-address\\images\\"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the upload folder
model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

OCR_LLM_MODEL = ImageOCRLLM()
GPT_LLM = GPTLLM()

DETECT_MODEL = GroundingDinoDetector(model=model)

@app.route('/upload', methods=['POST'])
async def upload_images():
    image_files = request.files.getlist('image')
    for image_file in image_files:
        print(image_file.filename)
        filepath = "D:\\content_detection\\source_code\\Image-to-address\\images\\" + image_file.filename
        image_file.save(filepath) # save locally
        image_exif_data = get_exif_data(filepath)
        store_sign_images = DETECT_MODEL.predict_billboards(filepath, image_exif_data)
        poi_json = []
        for image_frame in store_sign_images:
            try:
                ocr_values = OCR_LLM_MODEL.extract_text_frame_image(image_frame)
                json_of_point = GPT_LLM.get_poi_from_text(ocr_values)
                json_of_point["ocr_result"] = ocr_values
                json_of_point["file_name"] = filepath
                # if street_name:
                #     json_of_point["street_name"] = GPT_LLM.correct_street_name(
                #         street_name,
                #         "quận 8",
                #         "thành phố Hồ Chí Minh"
                #     )
                json_of_point['file_name'] = image_exif_data['File Name']
                json_of_point['lat'] = float(image_exif_data['Latitude'])
                json_of_point['Longitude'] = float(image_exif_data['Longitude'])
                json_of_point['created_date'] = image_exif_data['Date Created']
                json_of_point['modified_date'] = image_exif_data['Date Modified']
                result = await send_post_request(json_of_point)
                poi = json.loads(result)
                poi["street_no"] = json_of_point["street_no"]
                poi["street_name"] = json_of_point["street_name"]
                poi["district"] = json_of_point["district"]
                poi["city"] = json_of_point["city"]
                poi["ward"] = json_of_point["ward"]
                poi["category"] = json_of_point["category"]
                poi["product"] = json_of_point["product"]
                poi_json.append(poi)
            except Exception as error:
                print (str(error))
                print ("bad images")
                return jsonify({'message': 'bad image'}), 400
        return poi_json, 200
        

if __name__ == '__main__':

    # Ensure the upload folder exists
    app.run(debug=True)
