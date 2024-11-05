from pydantic import BaseModel, Field
from typing import Optional

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
