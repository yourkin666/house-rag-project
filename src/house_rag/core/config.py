"""
配置管理模块
"""
import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class Config:
    """应用配置类"""
    
    # Google API 配置
    GOOGLE_API_KEY: str = os.getenv('GOOGLE_API_KEY', '')
    
    # 数据库配置
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: str = os.getenv('DB_PORT', '5432')
    DB_NAME: str = os.getenv('DB_NAME', 'house_knowledge_base')
    DB_USER: str = os.getenv('DB_USER', 'houseuser')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', 'securepassword123')
    
    # 向量数据库配置
    COLLECTION_NAME: str = os.getenv('COLLECTION_NAME', 'langchain')
    
    # 应用配置
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    
    @property
    def database_url(self) -> str:
        """构建数据库连接URL"""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
    
    @property
    def pgvector_connection_string(self) -> str:
        """构建PGVector连接字符串"""
        return self.database_url
    
    def validate(self) -> None:
        """验证必需的配置项"""
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY is required but not set in environment variables")
        
        required_db_configs = [
            (self.DB_HOST, "DB_HOST"),
            (self.DB_PORT, "DB_PORT"),
            (self.DB_NAME, "DB_NAME"),
            (self.DB_USER, "DB_USER"),
            (self.DB_PASSWORD, "DB_PASSWORD")
        ]
        
        for value, name in required_db_configs:
            if not value:
                raise ValueError(f"{name} is required but not set in environment variables")


# 全局配置实例
config = Config()
