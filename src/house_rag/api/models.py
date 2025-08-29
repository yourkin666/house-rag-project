"""
API请求和响应模型定义
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """用户问题请求模型"""
    question: str
    max_results: Optional[int] = 3
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "我想找一个上海带游泳池的豪华别墅",
                "max_results": 3
            }
        }


class QuestionResponse(BaseModel):
    """问题回答响应模型"""
    answer: str
    retrieved_properties: List[dict]
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "根据您的需求，我找到了一套位于上海市浦东新区的豪华家庭别墅...",
                "retrieved_properties": [
                    {
                        "id": 1,
                        "title": "豪华家庭别墅",
                        "location": "上海市浦东新区",
                        "price": 1500.0
                    }
                ]
            }
        }


class PropertyRequest(BaseModel):
    """添加房源请求模型"""
    title: str = Field(..., description="房源标题", min_length=1, max_length=200)
    location: str = Field(..., description="房源位置", min_length=1, max_length=200)
    price: float = Field(..., description="房源价格（万元）", gt=0)
    description: str = Field(..., description="房源详细描述", min_length=10, max_length=2000)
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "现代简约公寓",
                "location": "上海市徐汇区",
                "price": 680.0,
                "description": "位于徐汇区核心地段的现代简约公寓，面积120平方米，三室两厅两卫，装修精美，家电齐全。小区环境优雅，交通便利，距离地铁站步行5分钟。周边有商场、学校、医院等配套设施。适合家庭居住或投资出租。"
            }
        }


class PropertyResponse(BaseModel):
    """添加房源响应模型"""
    success: bool
    message: str
    property_id: Optional[int] = None
    vector_generated: bool
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "房源已成功添加并完成向量化",
                "property_id": 15,
                "vector_generated": True
            }
        }