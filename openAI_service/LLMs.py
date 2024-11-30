from openai import OpenAI
from openAI_service.models import ShopInfo
from langchain.output_parsers import PydanticOutputParser
import json


class GPTLLM:
    
    def __init__(self) -> None:
        self.API_KEY = "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"
        self.openAI_client = OpenAI(api_key=self.API_KEY)
        self.GPT_MODEL = "gpt-3.5-turbo"
    
    def correct_street_name(self, street_name, district, city):
        system_context = """
                        {district} {city} có đường này không.
                        Nếu không hãy chỉnh sửa nó thành tên đường đúng và chỉ trả lời duy nhất bằng tên đường.
                """
        print (system_context.format(district=district, city=city))
        messages=[
            {
                "role": "system",
                "content": system_context.format(district=district, city=city)
            },
            {
                "role": "user",
                "content": street_name
            }
        ]
        config_dict = {
            "model": "gpt-4o",
            "messages": messages,
            "temperature": 1,
            "max_tokens": 2048,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        return_prompt = self._prompt(config_dict)
        print (return_prompt)
        return return_prompt
    
    def get_poi_from_text(self, ocr_values):
        messages=[
            {
               "role": "system",
            "content": """
            Bạn sẽ nhận được những đoạn text được trích xuất từ biển hiệu. 
            Bạn cần tổ chức và miêu tả thông tin rõ ràng, bao gồm:
            1. Tên cửa hàng (tên doanh nghiệp)
            2. Địa chỉ của cửa hàng (địa điểm)
            3. Các sản phẩm mà cửa hàng kinh doanh (liệt kê sản phẩm)
            4. Các danh mục kinh doanh (liệt kê danh mục sản phẩm, ví dụ như thời trang, đồ điện tử, thực phẩm,...).
            Mỗi mục cần được viết rõ ràng và chính xác. Ví dụ:
            - Tên cửa hàng: ABC Shop
            - Địa chỉ: 123 Đường XYZ, TP.HCM
            - Sản phẩm: Quần áo, giày dép, phụ kiện
            - Danh mục kinh doanh: Thời trang, phụ kiện
            Nếu không có thông tin, hãy trả về null cho từng trường không có giá trị.
            Ví dụ:
            - Tên cửa hàng: null
            - Địa chỉ: null
            - Sản phẩm: null
            - Danh mục kinh doanh: null

            """
            },
            {
                "role": "user",
                "content": ocr_values
            }
        ]
        config_dict = {
            "model": self.GPT_MODEL,
            "messages": messages,
            "temperature": 0.5,
            "max_tokens": 2048,
            "top_p": 1,
            "frequency_penalty": 0,
            "presence_penalty": 0
        }
        first_prompt = self._prompt(config_dict)
        pydantic_parser = PydanticOutputParser(pydantic_object=ShopInfo)
        messages.append({
            "role": "system",
            "content": f"Vui lòng trả lời theo định dạng sau: {pydantic_parser.get_format_instructions()}"
        })
        messages.append({
            "role": "user",
            "content": pydantic_parser.get_format_instructions()
        })
        print(first_prompt)

        final_answer = self._prompt(config_dict)
        print (final_answer)
        return json.loads(final_answer)
        
    def _prompt(self, config_dict):
        try:
            response = self.openAI_client.chat.completions.create(
                **config_dict
            )
            return response.choices[0].message.content
        except Exception:
            response = self.openAI_client.beta.chat.completions.parse(
                **config_dict
            )
            return response.choices[0].message.content            
