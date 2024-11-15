from pydantic import BaseModel, Field
from typing import Optional


FUNCTION_SCHEMA = {
    "name": "generate_response",
    "description": "Generate a structured response",
    "parameters": {
        "type": "object",
        "properties": {
            "shop_name": {
                "type": "string",
                "description": "Tên cửa hàng"
            },
            "address": {
                "type": "string",
                "description": "Địa chỉ của cửa hàng"
            },
            "product": {
                "type": "string",
                "description": "sản phẩm của cửa hàng"
            },
            "category": {
                "type": "string",
                "description": "danh mục kinh doanh của cửa hàng"                
            },
            "email": {
                "type": "string",
                "description": "email liên lạc của cửa hàng"
            },
            "phone_number": {
                "type": "string",
                "description": "số điện thoại liên lạc của cửa hàng"
            },
            "district": {
                "type": "string",
                "description": "quận trên địa chỉ của cửa hàng"
            },
            "street_no": {
                "type": "string",
                "description": "số nhà trên địa chỉ của cửa hàng"
            },
            "city": {
                "type": "string",
                "description": "thành phố trên địa chỉ của cửa hàng"
            },
            "ward": {
                "type": "string",
                "description": "phường trên địa chỉ của cửa hàng"
            }
        },
        "required": ["shop_name", "address"]
    }

}

class ShopInfo(BaseModel):
    shop_name: Optional[str] = Field(
        description="Tên của hàng"
    )
    address: Optional[str] = Field(
        description="Địa chỉ liên hệ"
    )
    phone_number: Optional[str] = Field(
        description="Số điện thoại liên hệ"
    )
    email: Optional[str] = Field(
        description="Email liên hệ của cửa hàng"
    )
    category: Optional[str] = Field(
        description="ngành nghề kinh doanh"
    )
    product: Optional[str] = Field(
        description="Sản phẩm cửa hàng kinh doanh"
    )
    district: Optional[str] = Field(
        description="quận trên địa chỉ của cửa hàng"
    )
    street_no: Optional[str] = Field(
        description="số nhà trên địa chỉ của cửa hàng"
    )
    street_name: Optional[str] = Field(
        description="tên đường trên địa chỉ cửa hàng"
    )

    city: Optional[str] = Field(
        description="thành phố trên địa chỉ của cửa hàng"
    )
    ward: Optional[str] = Field(
        description="phường trên địa chỉ của cửa hàng"
    )
