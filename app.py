#!/usr/bin/env python3
"""
房源 RAG 知识库 API 服务 - 主入口文件
为保持向后兼容性，此文件导入并启动重构后的应用
"""
from src.house_rag.api.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
