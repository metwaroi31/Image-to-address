from openai import OpenAI
from openAI_service.models import ShopInfo
from langchain.output_parsers import PydanticOutputParser
import json


class GPTLLM:
    
    def __init__(self) -> None:
        self.API_KEY = "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"
        self.openAI_client = OpenAI(api_key=self.API_KEY)
        self.GPT_MODEL = "gpt-3.5-turbo"
        
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
        messages.append({
            "role": "system",
            "content": first_prompt
        })
        messages.append({
            "role": "user",
            "content": """
                Cho tôi biết tất cả các đoạn đường ở quận 8 và thành phố Hồ Chí Minh.
            """
        })
        street_prompt = self._prompt(config_dict)
        pydantic_parser = PydanticOutputParser(pydantic_object=ShopInfo)
        messages.append({
            "role": "system",
            "content": f"Vui lòng trả lời theo định dạng sau: {pydantic_parser.get_format_instructions()}. Và chỉnh sửa tên đường theo danh sách sau {street_prompt}"
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
