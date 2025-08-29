#!/usr/bin/env python3
"""
混合搜索数据库迁移脚本

功能：
1. 执行数据库结构迁移（添加search_vector列和索引）
2. 为现有房源数据生成全文搜索向量
3. 验证迁移结果

使用方法：
- 确保 Docker 容器正在运行
- 在容器内运行：python -m house_rag.scripts.migrate_hybrid_search
- 或从宿主机运行：docker-compose exec app python -m house_rag.scripts.migrate_hybrid_search
"""

import sys
import os
import logging
from pathlib import Path
from sqlalchemy import text

# 添加项目路径到Python路径中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, project_root)

from house_rag.core.config import config
from house_rag.core.database import db_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_migration_prerequisites() -> bool:
    """检查迁移前置条件"""
    try:
        print("📋 检查迁移前置条件...")
        
        # 1. 检查数据库连接
        if not db_manager.test_connection():
            print("❌ 数据库连接失败")
            return False
        print("✅ 数据库连接正常")
        
        # 2. 检查PGVector扩展
        with db_manager.engine.connect() as conn:
            result = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")).fetchone()
            if not result:
                print("❌ PGVector扩展未安装")
                return False
        print("✅ PGVector扩展已安装")
        
        # 3. 检查properties表是否存在
        with db_manager.engine.connect() as conn:
            result = conn.execute(text("""
                SELECT 1 FROM information_schema.tables 
                WHERE table_name = 'properties' AND table_schema = 'public'
            """)).fetchone()
            if not result:
                print("❌ properties表不存在")
                return False
        print("✅ properties表存在")
        
        # 4. 检查现有房源数据
        total_count, _, _ = db_manager.get_properties_count()
        print(f"📊 发现 {total_count} 条房源记录")
        
        return True
        
    except Exception as e:
        print(f"❌ 前置条件检查失败: {e}")
        return False


def check_if_already_migrated() -> bool:
    """检查是否已经执行过迁移"""
    try:
        with db_manager.engine.connect() as conn:
            # 检查search_vector列是否存在
            result = conn.execute(text("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'properties' 
                AND column_name = 'search_vector'
                AND table_schema = 'public'
            """)).fetchone()
            
            if result:
                print("⚠️ 检测到已经执行过混合搜索迁移")
                
                # 检查有多少记录已经有search_vector
                count_result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(search_vector) as with_vector
                    FROM properties
                """)).fetchone()
                
                total = count_result[0]
                with_vector = count_result[1]
                
                print(f"📊 总房源数: {total}, 已生成搜索向量: {with_vector}")
                
                if total > 0 and with_vector == total:
                    print("✅ 所有房源都已有搜索向量，无需重复迁移")
                    return True
                elif with_vector > 0:
                    print("🔄 部分房源缺少搜索向量，将执行增量更新")
                    return False
                else:
                    print("🔄 需要为所有房源生成搜索向量")
                    return False
            
            return False
            
    except Exception as e:
        logger.error(f"检查迁移状态失败: {e}")
        return False


def execute_sql_migration() -> bool:
    """执行SQL数据库结构迁移"""
    try:
        print("🔧 开始执行数据库结构迁移...")
        
        # 读取迁移SQL文件
        migration_file = Path(project_root) / "database" / "add_fulltext_search.sql"
        if not migration_file.exists():
            print(f"❌ 迁移文件不存在: {migration_file}")
            return False
        
        with open(migration_file, 'r', encoding='utf-8') as f:
            migration_sql = f.read()
        
        # 执行迁移SQL
        with db_manager.engine.connect() as conn:
            trans = conn.begin()
            try:
                # 分割SQL语句并逐个执行
                sql_statements = [stmt.strip() for stmt in migration_sql.split(';') if stmt.strip() and not stmt.strip().startswith('/*')]
                
                for stmt in sql_statements:
                    if stmt and not stmt.startswith('--'):
                        logger.info(f"执行SQL: {stmt[:100]}...")
                        conn.execute(text(stmt))
                
                trans.commit()
                print("✅ 数据库结构迁移完成")
                return True
                
            except Exception as e:
                trans.rollback()
                print(f"❌ 数据库迁移失败: {e}")
                return False
                
    except Exception as e:
        print(f"❌ 读取迁移文件失败: {e}")
        return False


def update_search_vectors() -> bool:
    """为现有房源数据生成搜索向量"""
    try:
        print("🤖 开始生成搜索向量...")
        
        # 使用数据库管理器的重建方法
        updated_count = db_manager.rebuild_search_vectors()
        
        print(f"✅ 成功为 {updated_count} 条房源生成搜索向量")
        return True
        
    except Exception as e:
        print(f"❌ 生成搜索向量失败: {e}")
        return False


def verify_migration() -> bool:
    """验证迁移结果"""
    try:
        print("🔍 验证迁移结果...")
        
        with db_manager.engine.connect() as conn:
            # 1. 检查search_vector列
            result = conn.execute(text("""
                SELECT 1 FROM information_schema.columns
                WHERE table_name = 'properties' 
                AND column_name = 'search_vector'
                AND data_type = 'tsvector'
            """)).fetchone()
            
            if not result:
                print("❌ search_vector列验证失败")
                return False
            print("✅ search_vector列存在且类型正确")
            
            # 2. 检查索引
            result = conn.execute(text("""
                SELECT 1 FROM pg_indexes 
                WHERE tablename = 'properties' 
                AND indexname = 'idx_properties_search_vector'
            """)).fetchone()
            
            if not result:
                print("❌ 搜索索引验证失败")
                return False
            print("✅ 搜索索引存在")
            
            # 3. 检查函数
            result = conn.execute(text("""
                SELECT 1 FROM information_schema.routines 
                WHERE routine_name = 'generate_search_vector'
            """)).fetchone()
            
            if not result:
                print("❌ 搜索向量生成函数验证失败")
                return False
            print("✅ 搜索向量生成函数存在")
            
            # 4. 检查触发器
            result = conn.execute(text("""
                SELECT 1 FROM information_schema.triggers 
                WHERE trigger_name = 'trigger_update_search_vector'
            """)).fetchone()
            
            if not result:
                print("❌ 自动更新触发器验证失败")
                return False
            print("✅ 自动更新触发器存在")
            
            # 5. 统计数据
            result = conn.execute(text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(search_vector) as with_vector,
                    COUNT(CASE WHEN search_vector IS NOT NULL THEN 1 END) as non_null_vectors
                FROM properties
            """)).fetchone()
            
            total = result[0]
            with_vector = result[1]
            non_null_vectors = result[2]
            
            print(f"📊 统计结果:")
            print(f"   总房源数: {total}")
            print(f"   有搜索向量: {with_vector}")
            print(f"   非空搜索向量: {non_null_vectors}")
            
            if total > 0 and with_vector == total:
                print("✅ 所有房源都已生成搜索向量")
                
                # 6. 测试搜索功能
                test_result = conn.execute(text("""
                    SELECT COUNT(*) FROM properties 
                    WHERE search_vector @@ to_tsquery('simple', '房源')
                """)).fetchone()
                
                search_count = test_result[0]
                print(f"🔍 测试搜索 '房源': 找到 {search_count} 条记录")
                
                if search_count > 0:
                    print("✅ 全文搜索功能正常")
                    return True
                else:
                    print("⚠️ 全文搜索功能可能有问题")
                    return True  # 结构迁移成功，只是搜索结果为空
            else:
                print("⚠️ 部分房源缺少搜索向量")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 验证迁移结果失败: {e}")
        return False


def main():
    """主函数"""
    try:
        print("🚀 混合搜索数据库迁移脚本启动")
        print("=" * 60)
        
        # 1. 验证配置
        print("📋 验证配置...")
        config.validate()
        print("✅ 配置验证通过")
        
        # 2. 检查前置条件
        if not check_migration_prerequisites():
            print("❌ 前置条件检查失败，迁移中止")
            return False
        
        # 3. 检查是否已经迁移
        if check_if_already_migrated():
            print("🎉 迁移已完成，无需重复执行")
            return True
        
        # 4. 执行数据库结构迁移
        if not execute_sql_migration():
            print("❌ 数据库结构迁移失败")
            return False
        
        # 5. 生成搜索向量
        if not update_search_vectors():
            print("❌ 搜索向量生成失败")
            return False
        
        # 6. 验证迁移结果
        if not verify_migration():
            print("❌ 迁移验证失败")
            return False
        
        print("=" * 60)
        print("🎉 混合搜索迁移完成！")
        print("📈 现在您可以享受更精准的搜索体验了")
        print("\n🔧 下一步建议：")
        print("   1. 重启应用服务以加载新功能")
        print("   2. 测试一些具体的楼盘名称搜索")
        print("   3. 观察混合搜索的效果提升")
        
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了迁移过程")
        return False
    except Exception as e:
        print(f"❌ 迁移过程出现异常: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
