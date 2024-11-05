from transformers import AutoModel, AutoTokenizer
import torch
import ast

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
    
    def prompt_model(self, pixel_values, question):
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
