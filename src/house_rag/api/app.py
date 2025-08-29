"""
房源 RAG 知识库 API 服务
重构后的主应用文件
"""
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    QuestionRequest, QuestionResponse, HealthResponse,
    PropertyRequest, PropertyResponse, PropertyCountResponse
)
from ..core.config import config
from ..core.database import db_manager
from ..core.embeddings import rag_service

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI 应用实例
app = FastAPI(
    title="房源 RAG 知识库 API",
    description="一个基于检索增强生成(RAG)的房源信息查询API",
    version="2.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境中应该限制为特定域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """应用启动事件"""
    try:
        # 验证配置
        config.validate()
        logger.info("配置验证成功")
        
        # 测试数据库连接
        if not db_manager.test_connection():
            raise Exception("数据库连接测试失败")
        logger.info("数据库连接正常")
        
        logger.info("房源 RAG API 服务启动成功!")
    except Exception as e:
        logger.error(f"服务启动失败: {e}")
        raise


@app.get("/")
async def root():
    """根路径，返回API信息"""
    return {
        "message": "房源 RAG 知识库 API",
        "version": "2.0.0",
        "docs": "/docs",
        "status": "运行中"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查接口"""
    try:
        # 测试数据库连接
        db_connected = db_manager.test_connection()
        
        if db_connected:
            return HealthResponse(
                status="healthy",
                message="服务运行正常，数据库连接正常"
            )
        else:
            return HealthResponse(
                status="unhealthy",
                message="数据库连接异常"
            )
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="健康检查失败")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    智能房源查询接口
    
    接收用户的自然语言问题，通过RAG技术返回相关的房源推荐和详细回答
    """
    try:
        logger.info(f"收到问题查询: {request.question}")
        
        # 调用RAG服务进行查询
        result = rag_service.query_properties(
            question=request.question,
            max_results=request.max_results
        )
        
        logger.info("问题查询完成")
        return QuestionResponse(
            answer=result["answer"],
            retrieved_properties=result["retrieved_properties"]
        )
        
    except Exception as e:
        logger.error(f"问题查询失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@app.post("/append", response_model=PropertyResponse)
async def add_property(request: PropertyRequest):
    """
    添加新房源接口
    
    添加新的房源数据，自动进行向量化处理
    """
    try:
        logger.info(f"收到新房源添加请求: {request.title}")
        
        # 添加房源到数据库
        property_id = db_manager.add_property(
            title=request.title,
            location=request.location,
            price=request.price,
            description=request.description
        )
        
        # 尝试生成向量
        vector_generated = False
        try:
            vector_generated = rag_service.vectorize_property(
                property_id=property_id,
                title=request.title,
                location=request.location,
                price=request.price,
                description=request.description
            )
        except Exception as e:
            logger.warning(f"向量化失败，但房源已添加: {e}")
        
        message = "房源已成功添加"
        if vector_generated:
            message += "并完成向量化"
        else:
            message += "，但向量化失败，请稍后重试"
        
        logger.info(f"房源添加完成，ID: {property_id}, 向量化: {vector_generated}")
        return PropertyResponse(
            success=True,
            message=message,
            property_id=property_id,
            vector_generated=vector_generated
        )
        
    except Exception as e:
        logger.error(f"添加房源失败: {e}")
        raise HTTPException(status_code=500, detail=f"添加房源失败: {str(e)}")


@app.get("/properties/count", response_model=PropertyCountResponse)
async def get_properties_count():
    """
    获取房源数量统计接口
    
    返回总房源数、已向量化数量和待向量化数量
    """
    try:
        total, vectorized, pending = db_manager.get_properties_count()
        
        return PropertyCountResponse(
            total_properties=total,
            vectorized_properties=vectorized,
            pending_vectorization=pending
        )
        
    except Exception as e:
        logger.error(f"获取房源统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计失败: {str(e)}")


@app.get("/properties/{property_id}")
async def get_property(property_id: int):
    """
    获取指定房源详情接口
    """
    try:
        property_data = db_manager.get_property_by_id(property_id)
        
        if not property_data:
            raise HTTPException(status_code=404, detail="房源不存在")
        
        return property_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取房源详情失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取房源详情失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
