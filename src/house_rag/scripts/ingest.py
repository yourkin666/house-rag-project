#!/usr/bin/env python3
"""
房源数据向量化脚本 (重构版本)

此脚本的功能：
1. 从 PostgreSQL 数据库中读取还未向量化的房源数据
2. 使用 Google Gemini embedding 模型将房源描述转换为向量
3. 将生成的向量更新回数据库

使用方法：
- 确保 Docker 容器正在运行
- 在容器内运行：python -m house_rag.scripts.ingest
- 或从宿主机运行：docker-compose exec app python -m house_rag.scripts.ingest
"""

import sys
import os
import pandas as pd
from sqlalchemy import text
from typing import List

# 添加项目路径到Python路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from house_rag.core.config import config
from house_rag.core.database import db_manager
from house_rag.core.embeddings import rag_service

import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_properties_needing_vectorization() -> pd.DataFrame:
    """加载需要向量化的房源数据"""
    try:
        query = """
        SELECT id, title, location, price, description 
        FROM properties 
        WHERE description_embedding IS NULL 
        AND description IS NOT NULL 
        AND TRIM(description) != ''
        ORDER BY id
        """
        
        with db_manager.engine.connect() as conn:
            df = pd.read_sql_query(text(query), conn)
        
        logger.info(f"发现 {len(df)} 条待处理的房源数据")
        return df
    except Exception as e:
        logger.error(f"加载房源数据失败: {e}")
        raise


def prepare_texts_for_vectorization(df: pd.DataFrame) -> List[str]:
    """准备用于向量化的文本内容"""
    texts = []
    for _, row in df.iterrows():
        text = f"房源：{row['title']}。位于 {row['location']}，价格 {row['price']}万元。{row['description']}"
        texts.append(text)
    
    logger.info(f"准备了 {len(texts)} 条文本内容用于向量化")
    return texts


def update_database_with_vectors(df: pd.DataFrame, embeddings: List[List[float]]) -> None:
    """将生成的向量更新到数据库"""
    success_count = 0
    
    try:
        with db_manager.engine.connect() as conn:
            trans = conn.begin()
            try:
                for i, (_, row) in enumerate(df.iterrows()):
                    property_id = row['id']
                    embedding = embeddings[i]
                    
                    # 更新数据库
                    result = conn.execute(
                        text("UPDATE properties SET description_embedding = :embedding WHERE id = :id"),
                        {"embedding": embedding, "id": property_id}
                    )
                    
                    if result.rowcount > 0:
                        success_count += 1
                    
                trans.commit()
                logger.info(f"成功更新 {success_count} 条房源的向量数据")
                
            except Exception:
                trans.rollback()
                raise
                
    except Exception as e:
        logger.error(f"更新数据库失败: {e}")
        raise


def add_to_vector_store(df: pd.DataFrame) -> None:
    """将房源数据添加到向量存储"""
    try:
        for _, row in df.iterrows():
            property_data = {
                'id': row['id'],
                'title': row['title'],
                'location': row['location'],
                'price': float(row['price']) if row['price'] else 0.0,
                'description': row['description']
            }
            
            try:
                rag_service.add_document_to_vectorstore(property_data)
            except Exception as e:
                logger.warning(f"添加房源 {row['id']} 到向量存储失败: {e}")
                
        logger.info("向量存储更新完成")
    except Exception as e:
        logger.error(f"更新向量存储失败: {e}")
        raise


def main():
    """主函数"""
    try:
        print("🏠 房源数据向量化脚本启动")
        print("=" * 50)
        
        # 验证配置
        print("📋 正在验证配置...")
        config.validate()
        print("✅ 配置验证通过")
        
        # 测试数据库连接
        print("🔌 正在测试数据库连接...")
        if not db_manager.test_connection():
            raise Exception("数据库连接失败")
        print(f"✅ 成功连接到数据库: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
        
        # 加载待处理的房源数据
        print("📊 正在加载待处理的房源数据...")
        df = load_properties_needing_vectorization()
        
        if len(df) == 0:
            print("🎉 所有房源数据都已完成向量化！")
            return
        
        print(f"📝 找到 {len(df)} 条待处理房源")
        
        # 准备文本内容
        print("📝 正在准备向量化文本内容...")
        texts = prepare_texts_for_vectorization(df)
        
        # 生成向量
        print("🤖 正在使用 Google Gemini 生成向量...")
        print(f"📝 处理 {len(texts)} 条文本内容")
        embeddings = rag_service.generate_embeddings_batch(texts)
        print(f"✅ 成功生成 {len(embeddings)} 个向量")
        print(f"📐 向量维度: {len(embeddings[0]) if embeddings else 0}")
        
        # 更新数据库
        print("💾 正在更新数据库...")
        update_database_with_vectors(df, embeddings)
        
        # 更新向量存储
        print("🗂️ 正在更新向量存储...")
        add_to_vector_store(df)
        
        print("=" * 50)
        print("🎉 向量化处理完成！")
        print(f"✨ 成功处理 {len(df)} 条房源数据")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了处理过程")
    except Exception as e:
        print(f"❌ 向量化处理失败: {e}")
        logger.error(f"向量化处理失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
