#!/usr/bin/env python3
"""
混合搜索功能测试脚本

功能：
1. 测试纯向量搜索和混合搜索的对比
2. 验证RRF算法的融合效果
3. 测试不同类型查询的改进情况
4. 生成性能和效果报告

使用方法：
- 确保已完成混合搜索迁移
- python test_hybrid_search.py
"""

import sys
import os
import time
import logging
from typing import List, Dict, Any
from pathlib import Path

# 添加项目路径
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from src.house_rag.core.embeddings import rag_service
from src.house_rag.core.database import db_manager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class HybridSearchTester:
    """混合搜索测试器"""
    
    def __init__(self):
        self.test_queries = [
            # 具体楼盘名称测试（应该更有利于关键词搜索）
            {
                "query": "汤臣一品",
                "type": "specific_name",
                "description": "具体楼盘名称"
            },
            {
                "query": "绿城",
                "type": "brand_name", 
                "description": "品牌/开发商名称"
            },
            # 语义查询测试（应该更有利于向量搜索）
            {
                "query": "适合家庭居住的安静房子",
                "type": "semantic",
                "description": "语义理解查询"
            },
            {
                "query": "性价比高的学区房",
                "type": "semantic",
                "description": "复合概念查询"
            },
            # 混合查询测试（应该混合搜索表现更好）
            {
                "query": "浦东新区豪华别墅",
                "type": "hybrid",
                "description": "地理位置+属性描述"
            },
            {
                "query": "上海市中心精装修公寓",
                "type": "hybrid", 
                "description": "位置+装修+类型"
            },
            # 长尾查询测试
            {
                "query": "带花园停车位的独栋别墅价格1500万以内",
                "type": "complex",
                "description": "复杂长尾查询"
            }
        ]
        
        self.results = []
    
    def test_single_query(self, query: str, query_type: str, description: str) -> Dict[str, Any]:
        """测试单个查询的混合搜索效果"""
        print(f"\n🔍 测试查询: '{query}' ({description})")
        print("-" * 50)
        
        result = {
            "query": query,
            "type": query_type,
            "description": description
        }
        
        try:
            # 1. 测试纯向量搜索（禁用混合搜索）
            print("📊 纯向量搜索:")
            start_time = time.time()
            rag_service.hybrid_search_enabled = False
            vector_response = rag_service.ask_question(query, max_results=5)
            vector_time = time.time() - start_time
            
            vector_properties = vector_response.get('retrieved_properties', [])
            print(f"   找到 {len(vector_properties)} 个结果，用时 {vector_time:.2f}s")
            if vector_properties:
                print(f"   Top结果: {vector_properties[0].get('title', 'N/A')}")
            
            # 2. 测试混合搜索
            print("🔀 混合搜索:")
            start_time = time.time()
            rag_service.hybrid_search_enabled = True
            hybrid_response = rag_service.ask_question(query, max_results=5)
            hybrid_time = time.time() - start_time
            
            hybrid_properties = hybrid_response.get('retrieved_properties', [])
            print(f"   找到 {len(hybrid_properties)} 个结果，用时 {hybrid_time:.2f}s")
            if hybrid_properties:
                print(f"   Top结果: {hybrid_properties[0].get('title', 'N/A')}")
            
            # 3. 测试全文搜索单独效果
            print("📝 纯全文搜索:")
            start_time = time.time()
            fulltext_results = db_manager.fulltext_search(query, limit=5)
            fulltext_time = time.time() - start_time
            
            print(f"   找到 {len(fulltext_results)} 个结果，用时 {fulltext_time:.2f}s")
            if fulltext_results:
                # 获取第一个结果的详细信息
                first_prop = db_manager.get_property_by_id(fulltext_results[0][0])
                if first_prop:
                    print(f"   Top结果: {first_prop.get('title', 'N/A')}")
            
            # 4. 分析结果差异
            vector_ids = set(prop.get('id') for prop in vector_properties)
            hybrid_ids = set(prop.get('id') for prop in hybrid_properties)
            fulltext_ids = set(result[0] for result in fulltext_results)
            
            only_in_hybrid = hybrid_ids - vector_ids
            only_in_vector = vector_ids - hybrid_ids
            in_both = vector_ids & hybrid_ids
            
            print(f"\n📊 结果分析:")
            print(f"   向量搜索独有: {len(only_in_vector)} 个")
            print(f"   混合搜索独有: {len(only_in_hybrid)} 个")  
            print(f"   两者都有: {len(in_both)} 个")
            print(f"   全文搜索结果: {len(fulltext_ids)} 个")
            
            # 保存结果
            result.update({
                "vector_count": len(vector_properties),
                "hybrid_count": len(hybrid_properties),
                "fulltext_count": len(fulltext_results),
                "vector_time": vector_time,
                "hybrid_time": hybrid_time,
                "fulltext_time": fulltext_time,
                "only_in_hybrid": len(only_in_hybrid),
                "only_in_vector": len(only_in_vector),
                "in_both": len(in_both),
                "vector_top_result": vector_properties[0].get('title') if vector_properties else None,
                "hybrid_top_result": hybrid_properties[0].get('title') if hybrid_properties else None,
                "vector_properties": vector_properties,
                "hybrid_properties": hybrid_properties
            })
            
        except Exception as e:
            print(f"❌ 测试查询失败: {e}")
            result["error"] = str(e)
        
        return result
    
    def test_rrf_algorithm(self):
        """测试RRF算法的融合效果"""
        print(f"\n🧮 测试 RRF 融合算法")
        print("=" * 50)
        
        # 模拟两个不同的搜索结果
        vector_results = [(1, 0.95), (2, 0.85), (3, 0.75), (5, 0.65)]
        fulltext_results = [(4, 0.9), (1, 0.8), (6, 0.7), (2, 0.6)]
        
        print("向量搜索结果:", vector_results)
        print("全文搜索结果:", fulltext_results)
        
        # 使用RRF融合
        hybrid_results = rag_service.rrf_fusion.fuse_rankings(vector_results, fulltext_results, max_results=6)
        
        print("\nRRF融合结果:")
        for i, result in enumerate(hybrid_results, 1):
            print(f"  {i}. 房源ID:{result.property_id}, "
                  f"融合分数:{result.final_score:.4f}, "
                  f"向量分数:{result.vector_score:.2f}, "
                  f"全文分数:{result.fulltext_score:.2f}")
        
        # 分析融合效果
        print(f"\n融合分析:")
        print(f"  房源1在向量搜索排名1，全文搜索排名2，融合排名: {next((i+1 for i, r in enumerate(hybrid_results) if r.property_id == 1), 'N/A')}")
        print(f"  房源4仅在全文搜索出现(排名1)，融合排名: {next((i+1 for i, r in enumerate(hybrid_results) if r.property_id == 4), 'N/A')}")
        print(f"  房源5仅在向量搜索出现(排名4)，融合排名: {next((i+1 for i, r in enumerate(hybrid_results) if r.property_id == 5), 'N/A')}")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 开始混合搜索功能测试")
        print("=" * 60)
        
        # 检查前置条件
        if not self.check_prerequisites():
            print("❌ 前置条件检查失败，测试中止")
            return
        
        # 测试RRF算法
        self.test_rrf_algorithm()
        
        # 测试各种查询
        for test_case in self.test_queries:
            result = self.test_single_query(
                test_case["query"],
                test_case["type"], 
                test_case["description"]
            )
            self.results.append(result)
        
        # 生成报告
        self.generate_report()
    
    def check_prerequisites(self) -> bool:
        """检查测试前置条件"""
        try:
            print("📋 检查测试前置条件...")
            
            # 检查数据库连接
            if not db_manager.test_connection():
                print("❌ 数据库连接失败")
                return False
            
            # 检查混合搜索功能是否可用
            with db_manager.engine.connect() as conn:
                from sqlalchemy import text
                result = conn.execute(text("""
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name = 'properties' 
                    AND column_name = 'search_vector'
                """)).fetchone()
                
                if not result:
                    print("❌ 混合搜索功能未安装，请先运行迁移脚本")
                    return False
            
            # 检查房源数据
            total_count, _, _ = db_manager.get_properties_count()
            if total_count == 0:
                print("❌ 没有房源数据可测试")
                return False
            
            print(f"✅ 前置条件检查通过，找到 {total_count} 条房源数据")
            return True
            
        except Exception as e:
            print(f"❌ 前置条件检查失败: {e}")
            return False
    
    def generate_report(self):
        """生成测试报告"""
        print(f"\n📊 混合搜索测试报告")
        print("=" * 60)
        
        if not self.results:
            print("❌ 没有测试结果")
            return
        
        # 统计各类查询的改进情况
        improvements = {
            "specific_name": [],
            "brand_name": [],
            "semantic": [],
            "hybrid": [],
            "complex": []
        }
        
        for result in self.results:
            if "error" in result:
                continue
                
            query_type = result["type"]
            
            # 计算改进指标
            vector_count = result.get("vector_count", 0)
            hybrid_count = result.get("hybrid_count", 0)
            only_in_hybrid = result.get("only_in_hybrid", 0)
            
            improvement = {
                "query": result["query"],
                "vector_count": vector_count,
                "hybrid_count": hybrid_count,
                "new_results": only_in_hybrid,
                "time_diff": result.get("hybrid_time", 0) - result.get("vector_time", 0)
            }
            
            if query_type in improvements:
                improvements[query_type].append(improvement)
        
        # 输出各类查询的表现
        for query_type, results in improvements.items():
            if not results:
                continue
                
            print(f"\n🎯 {query_type.upper()} 查询类型:")
            avg_new_results = sum(r["new_results"] for r in results) / len(results)
            avg_time_diff = sum(r["time_diff"] for r in results) / len(results)
            
            print(f"   平均新增结果: {avg_new_results:.1f} 个")
            print(f"   平均时间差异: {avg_time_diff*1000:.1f}ms")
            
            for result in results:
                status = "📈" if result["new_results"] > 0 else "➖"
                print(f"   {status} '{result['query'][:30]}...' 新增{result['new_results']}个结果")
        
        # 总体统计
        total_tests = len([r for r in self.results if "error" not in r])
        improved_tests = len([r for r in self.results if r.get("only_in_hybrid", 0) > 0])
        
        print(f"\n📈 总体表现:")
        print(f"   测试查询数: {total_tests}")
        print(f"   有改进的查询: {improved_tests}")
        print(f"   改进率: {improved_tests/total_tests*100:.1f}%" if total_tests > 0 else "N/A")
        
        # 混合搜索统计信息
        if hasattr(rag_service, 'hybrid_search_stats'):
            stats = rag_service.hybrid_search_stats
            print(f"\n🔀 混合搜索统计:")
            print(f"   总搜索次数: {stats.get('total_hybrid_searches', 0)}")
            print(f"   向量搜索回退: {stats.get('vector_only_fallbacks', 0)}")
            print(f"   全文搜索贡献: {stats.get('fulltext_contributions', 0)}")
        
        print(f"\n✨ 测试完成！混合搜索功能{' 正常工作' if improved_tests > 0 else ' 可能需要调优'}")


def main():
    """主函数"""
    tester = HybridSearchTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
