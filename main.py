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
from apis.map import send_post_request
from vintern_llava.models_for_prompt import ShopInfo
from vintern_llava.LLMs import ImageOCRLLM
from langchain.output_parsers import PydanticOutputParser
import torch
import glob
import asyncio
LLM_MODEL = ImageOCRLLM()

model = load_model(
    "groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)

TEXT_PROMPT = "billboard . sign . advertisement"
BOX_TRESHOLD = 0.20
TEXT_TRESHOLD = 0.20
pydantic_parser = PydanticOutputParser(pydantic_object=ShopInfo)
format_instructions = pydantic_parser.get_format_instructions()

while True:
    # TODO: For each image Get the following
    # Get multiple addresses and shop names
    # Get lat/lng
    # save some information for training
    # integrate with MapAPI
    JSON_to_csv = []
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
        pydantic_parser = PydanticOutputParser(pydantic_object=ShopInfo)
        format_instructions = pydantic_parser.get_format_instructions()
        question = '<image>\n' + format_instructions
        for image_frame in store_sign_images:
            pixel_values = from_tensor_to_pixels(image_frame, max_num=6).to(torch.bfloat16).cuda()
            json_of_point = {}
            json_of_point['file_name'] = image_exif_data['File Name']
            json_of_point['lat'] = float(image_exif_data['Latitude'])
            json_of_point['Longitude'] = float(image_exif_data['Longitude'])
            json_of_point['created_date'] = image_exif_data['Date Created']
            json_of_point['modified_date'] = image_exif_data['Date Modified']
            # json_of_point['shop_name'] = None
            # json_of_point['address'] = None
            try:
                shop_json = LLM_MODEL.prompt_model(pixel_values, question)
                json_of_point['shop_name'] = shop_json['shop_name']
                json_of_point['address'] = shop_json['address']
                json_of_point['category'] = shop_json['category']
                json_of_point['product'] = shop_json['product']
                print (shop_json)
                asyncio.run(send_post_request(json_of_point))
            except Exception as error:
                print (str(error))
                print ("bad images")
            JSON_to_csv.append(json_of_point)
            
    write_metadata_to_csv(JSON_to_csv, "report.csv")
    break
    
