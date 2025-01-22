from groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    crop_image,
    annotate
)
from image_utils.read_metadata import (
    write_metadata_to_csv,
    read_exif_with_exifread
)
from image_utils.video_utils import (
    extract_single_frame,
    _extract_gps_data,
)
from gps_utils.gps_handler import GPSHandler
from gps_utils.poi_optimizer import POIOptimizationProcessor
from openAI_service.LLMs import GPTLLM
from vintern_llava.LLMs import ImageOCRLLM
from apis.map import send_post_request
import asyncio
import glob
from groundingdino.detector import GroundingDinoDetector
import csv
from datetime import datetime


model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

OCR_LLM_MODEL = ImageOCRLLM()
GPT_LLM = GPTLLM()

DETECT_MODEL = GroundingDinoDetector(model=model)

video_file = "images_video/GH011305.MP4"
result = _extract_gps_data(video_file)
gps_handler = GPSHandler(result)
gpx_gps_data = gps_handler.gps_data
start_time = gpx_gps_data[0]['time']
date_format = "%Y-%m-%d %H:%M:%S"

# Interpolate GPS and create frames
gps_data = gps_handler.interpolate_gps()
gps_data = gps_handler.correct_gps_with_route_and_geo_api()
gps_data = gps_handler.create_gps_for_frames()

# Distance problem and 
number_5_poi_optimizer = POIOptimizationProcessor(
    gps_data
)
number_5_poi_optimizer.choose_poi_for_optimizing()
results_for_poi = number_5_poi_optimizer.find_optimal_position()
print (results_for_poi)
keys = ["shop_name","address","phone_number","email","category","product","district","street_no","street_name","city","ward","ocr_result","file_name", "lat", "Latitude", "Longitude"]
with open("report10-1.csv", mode='a', newline='', encoding='utf-8') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=keys)
    writer.writeheader()

    for i in range(len(results_for_poi)):
        time_to_extract = gpx_gps_data[i]['time']
        gps_datetime = datetime.strptime(time_to_extract, date_format)
        start_datetime = datetime.strptime(start_time, date_format)
        difference = gps_datetime - start_datetime
        file_name = extract_single_frame(
            video_file,
            results_for_poi[i],
            difference
        )
        image_exif_data = read_exif_with_exifread(file_name)
        store_sign_images = DETECT_MODEL.predict_billboards(file_name, image_exif_data)
        for image_frame in store_sign_images:
            try:
                        ocr_values = OCR_LLM_MODEL.extract_text_frame_image(image_frame)
                        json_of_point = GPT_LLM.get_poi_from_text(ocr_values)
                        street_name = json_of_point["street_name"]
                        json_of_point["ocr_result"] = ocr_values
                        json_of_point["file_name"] = file_name
                        json_of_point["Longitude"] = results_for_poi[i]['poi_lon']
                        json_of_point["lat"] = results_for_poi[i]['poi_lat']
                        json_of_point["file_name"] = image_exif_data["FileName"]
                        if street_name:
                            json_of_point["street_name"] = GPT_LLM.correct_street_name(
                                street_name,
                                "quận 7",
                                "thành phố Hồ Chí Minh"
                            )
                        writer.writerow(json_of_point)
                        result = asyncio.run(send_post_request(json_of_point))
                        print(json_of_point)
            except Exception as error:
                        print (str(error))
                        print ("bad images")
