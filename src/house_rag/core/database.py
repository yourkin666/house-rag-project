"""
数据库连接和操作模块
"""
import logging
from typing import List, Tuple, Optional
from sqlalchemy import create_engine, text, Engine
from sqlalchemy.exc import SQLAlchemyError

from .config import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self._connect()
    
    def _connect(self) -> None:
        """建立数据库连接"""
        try:
            self.engine = create_engine(
                config.database_url,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                echo=config.DEBUG
            )
            logger.info(f"成功连接到数据库: {config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}")
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            raise
    
    def test_connection(self) -> bool:
        """测试数据库连接"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("SELECT 1")).fetchone()
                return result[0] == 1
        except Exception as e:
            logger.error(f"数据库连接测试失败: {e}")
            return False
    
    def get_properties_count(self) -> Tuple[int, int, int]:
        """
        获取房源数量统计
        返回: (总数, 已向量化数量, 待向量化数量)
        """
        try:
            with self.engine.connect() as conn:
                # 总房源数
                total_result = conn.execute(text("SELECT COUNT(*) FROM properties")).fetchone()
                total_count = total_result[0]
                
                # 已向量化房源数
                vectorized_result = conn.execute(
                    text("SELECT COUNT(*) FROM properties WHERE description_embedding IS NOT NULL")
                ).fetchone()
                vectorized_count = vectorized_result[0]
                
                # 待向量化房源数
                pending_count = total_count - vectorized_count
                
                return total_count, vectorized_count, pending_count
        except SQLAlchemyError as e:
            logger.error(f"获取房源统计失败: {e}")
            raise
    
    def add_property(self, title: str, location: str, price: float, description: str) -> int:
        """
        添加新房源到数据库
        返回: 新插入房源的ID
        """
        try:
            with self.engine.connect() as conn:
                # 开始事务
                trans = conn.begin()
                try:
                    result = conn.execute(
                        text("""
                            INSERT INTO properties (title, location, price, description)
                            VALUES (:title, :location, :price, :description)
                            RETURNING id
                        """),
                        {
                            'title': title,
                            'location': location,
                            'price': price,
                            'description': description
                        }
                    )
                    property_id = result.fetchone()[0]
                    trans.commit()
                    logger.info(f"成功添加房源，ID: {property_id}")
                    return property_id
                except Exception:
                    trans.rollback()
                    raise
        except SQLAlchemyError as e:
            logger.error(f"添加房源失败: {e}")
            raise
    
    def update_property_embedding(self, property_id: int, embedding: List[float]) -> bool:
        """
        更新房源的向量数据
        返回: 是否更新成功
        """
        try:
            with self.engine.connect() as conn:
                # 开始事务
                trans = conn.begin()
                try:
                    result = conn.execute(
                        text("""
                            UPDATE properties 
                            SET description_embedding = :embedding
                            WHERE id = :property_id
                        """),
                        {
                            'embedding': embedding,
                            'property_id': property_id
                        }
                    )
                    trans.commit()
                    success = result.rowcount > 0
                    if success:
                        logger.info(f"成功更新房源 {property_id} 的向量数据")
                    else:
                        logger.warning(f"未找到ID为 {property_id} 的房源")
                    return success
                except Exception:
                    trans.rollback()
                    raise
        except SQLAlchemyError as e:
            logger.error(f"更新房源向量失败: {e}")
            raise
    
    def get_property_by_id(self, property_id: int) -> Optional[dict]:
        """
        根据ID获取房源信息
        返回: 房源字典或None
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(
                    text("""
                        SELECT id, title, location, price, description, created_at, updated_at
                        FROM properties 
                        WHERE id = :property_id
                    """),
                    {'property_id': property_id}
                ).fetchone()
                
                if result:
                    return {
                        'id': result[0],
                        'title': result[1],
                        'location': result[2],
                        'price': float(result[3]) if result[3] else None,
                        'description': result[4],
                        'created_at': result[5],
                        'updated_at': result[6]
                    }
                return None
        except SQLAlchemyError as e:
            logger.error(f"获取房源信息失败: {e}")
            raise
    
    def get_properties_by_ids(self, property_ids: List[int]) -> List[dict]:
        """
        根据ID列表批量获取房源信息
        返回: 房源字典列表
        """
        if not property_ids:
            return []
            
        try:
            with self.engine.connect() as conn:
                # 构建 IN 子句的参数
                placeholders = ','.join([f':id_{i}' for i in range(len(property_ids))])
                params = {f'id_{i}': property_ids[i] for i in range(len(property_ids))}
                
                results = conn.execute(
                    text(f"""
                        SELECT id, title, location, price, description, created_at, updated_at
                        FROM properties 
                        WHERE id IN ({placeholders})
                        ORDER BY id
                    """),
                    params
                ).fetchall()
                
                return [
                    {
                        'id': row[0],
                        'title': row[1],
                        'location': row[2],
                        'price': float(row[3]) if row[3] else None,
                        'description': row[4],
                        'created_at': row[5],
                        'updated_at': row[6]
                    }
                    for row in results
                ]
        except SQLAlchemyError as e:
            logger.error(f"批量获取房源信息失败: {e}")
            raise
    
    def fulltext_search(self, query: str, limit: int = 20, min_rank: float = 0.01) -> List[Tuple[int, float]]:
        """
        执行全文搜索
        
        Args:
            query: 搜索查询字符串
            limit: 返回结果数量限制
            min_rank: 最小相关性分数阈值
            
        Returns:
            List[Tuple[int, float]]: [(property_id, relevance_score), ...]
        """
        try:
            with self.engine.connect() as conn:
                # 清理查询字符串，移除特殊字符，用 & 连接多个词
                cleaned_query = self._clean_fulltext_query(query)
                if not cleaned_query:
                    return []
                
                results = conn.execute(
                    text("""
                        SELECT 
                            id,
                            ts_rank(search_vector, to_tsquery('simple', :query)) as rank
                        FROM properties 
                        WHERE search_vector @@ to_tsquery('simple', :query)
                        AND ts_rank(search_vector, to_tsquery('simple', :query)) >= :min_rank
                        ORDER BY rank DESC, id ASC
                        LIMIT :limit
                    """),
                    {
                        'query': cleaned_query,
                        'min_rank': min_rank,
                        'limit': limit
                    }
                ).fetchall()
                
                return [(row[0], float(row[1])) for row in results]
        except SQLAlchemyError as e:
            logger.error(f"全文搜索失败: {e}")
            # 如果全文搜索失败，返回空结果而不是抛出异常，确保混合搜索的鲁棒性
            logger.warning("全文搜索失败，将只使用向量搜索")
            return []
    
    def _clean_fulltext_query(self, query: str) -> str:
        """
        清理和预处理全文搜索查询字符串
        
        Args:
            query: 原始查询字符串
            
        Returns:
            str: 清理后的查询字符串，适用于 PostgreSQL 的 to_tsquery
        """
        if not query or not query.strip():
            return ""
        
        # 移除特殊字符，只保留字母、数字、中文字符和空格
        import re
        cleaned = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)
        
        # 将多个空格替换为单个空格，并去除首尾空格
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        if not cleaned:
            return ""
        
        # 将空格分隔的词用 & 连接（AND 操作）
        words = cleaned.split()
        if len(words) == 1:
            # 单个词，添加前缀匹配支持
            return f"{words[0]}:*"
        else:
            # 多个词，用 & 连接
            return ' & '.join([f"{word}:*" for word in words])
    
    def rebuild_search_vectors(self) -> int:
        """
        重建所有房源的搜索向量
        返回: 更新的记录数
        """
        try:
            with self.engine.connect() as conn:
                trans = conn.begin()
                try:
                    result = conn.execute(
                        text("""
                            UPDATE properties 
                            SET search_vector = generate_search_vector(title, location, description)
                            WHERE title IS NOT NULL OR location IS NOT NULL OR description IS NOT NULL
                        """)
                    )
                    updated_count = result.rowcount
                    trans.commit()
                    logger.info(f"成功重建 {updated_count} 条房源的搜索向量")
                    return updated_count
                except Exception:
                    trans.rollback()
                    raise
        except SQLAlchemyError as e:
            logger.error(f"重建搜索向量失败: {e}")
            raise


# 全局数据库管理器实例
db_manager = DatabaseManager()
