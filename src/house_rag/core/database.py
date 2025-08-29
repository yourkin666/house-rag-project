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


# 全局数据库管理器实例
db_manager = DatabaseManager()
