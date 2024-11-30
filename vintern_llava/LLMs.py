from transformers import AutoModel, AutoTokenizer
from langchain.output_parsers import PydanticOutputParser
from apis.map import send_post_request
from image_utils.image_processor import load_image_as_pixels
from image_utils.image_processor import from_tensor_to_pixels
from vintern_llava.models_for_prompt import ShopInfo
import torch
import ast
import asyncio

class ImageOCRLLM:
    
    def __init__(self) -> None:
        self.MODEL_NAME = "5CD-AI/Vintern-1B-v2"
        self.model = AutoModel.from_pretrained(
            self.MODEL_NAME,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).eval().cuda() 
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            trust_remote_code=True,
            use_fast=False
        )
        self.pydantic_parser = PydanticOutputParser(pydantic_object=ShopInfo)
        self.format_instructions = self.pydantic_parser.get_format_instructions()
        self.question = '<image>\n' + self.format_instructions
    
    def _prompt_model(self, pixel_values, question):
        generation_config = dict(max_new_tokens= 512, do_sample=False, num_beams = 3, repetition_penalty=3.5)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config)
        return ast.literal_eval(response)
    
    def prompt_chat_model(self, pixel_values):
        generation_config = dict(max_new_tokens= 512, do_sample=False, num_beams = 3, repetition_penalty=3.5)
        question = '<image>\n' + "Mô tả chi tiết tấm hình"

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config)
        return response
    
    def extract_text_images(self, file_name):
        pixel_values = load_image_as_pixels(file_name, max_num=6).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens= 1024, do_sample=False, num_beams = 3, repetition_penalty=3.5)
        question = '<image>\n' + "trích xuất tất cả các chữ và in ra các dòng"
        generation_config = dict(max_new_tokens= 512, do_sample=False, num_beams = 3, repetition_penalty=3.5)

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config)
        return response
    
    def extract_text_frame_image(self, image_frame):
        pixel_values = from_tensor_to_pixels(image_frame, max_num=6).to(torch.bfloat16).cuda()
        generation_config = dict(max_new_tokens= 1024, do_sample=False, num_beams = 3, repetition_penalty=3.5)
        question = '<image>\n' + "trích xuất tất cả các chữ và in ra các dòng"
        generation_config = dict(max_new_tokens= 512, do_sample=False, num_beams = 3, repetition_penalty=3.5)
        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config)
        return response

    
    def send_poi_info_to_db(self, image_frame, image_exif_data):
        json_of_point = {}
        pixel_values = from_tensor_to_pixels(image_frame, max_num=6).to(torch.bfloat16).cuda()
        json_of_point['file_name'] = image_exif_data['File Name']
        json_of_point['lat'] = float(image_exif_data['Latitude'])
        json_of_point['Longitude'] = float(image_exif_data['Longitude'])
        json_of_point['created_date'] = image_exif_data['Date Created']
        json_of_point['modified_date'] = image_exif_data['Date Modified']
        try:
            shop_json = self._prompt_model(pixel_values, self.question)
            json_of_point['shop_name'] = shop_json['shop_name']
            json_of_point['address'] = shop_json['address']
            json_of_point['category'] = shop_json['category']
            json_of_point['product'] = shop_json['product']
            asyncio.run(send_post_request(json_of_point))
        except Exception as error:
            print (str(error))
            print ("bad images")
