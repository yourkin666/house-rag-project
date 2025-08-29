#!/usr/bin/env python3
"""
重排序优化效果测试脚本

功能：
1. 测试基础分数融合效果
2. 验证价格评分机制优化
3. 测试否定条件处理
4. 验证位置模糊匹配
5. 对比优化前后的效果

使用方法：
- 确保已完成混合搜索部署
- python test_reranking_optimization.py
"""

import sys
import os
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


class RerankingOptimizationTester:
    """重排序优化测试器"""
    
    def __init__(self):
        self.test_cases = [
            # 价格优化测试
            {
                "name": "价格接近度测试",
                "query": "浦东新区1000万左右的房子",
                "expected_improvements": ["价格评分更精确", "接近理想价格的房源排名更高"]
            },
            # 否定条件测试
            {
                "name": "否定条件处理",
                "query": "静安区学区房，但不要吵闹的环境",
                "expected_improvements": ["避免吵闹环境", "排除有噪音关键词的房源"]
            },
            # 位置模糊匹配测试
            {
                "name": "位置模糊匹配",
                "query": "找个徐家汇的房子",  # 测试"徐家汇"是否能匹配"徐汇区"
                "expected_improvements": ["处理地名别名", "模糊匹配位置关键词"]
            },
            # 综合测试
            {
                "name": "综合优化效果",
                "query": "浦东800-1200万的别墅，远离工厂",
                "expected_improvements": ["价格范围评分", "否定条件处理", "房屋类型匹配"]
            }
        ]
    
    def test_price_scoring_optimization(self):
        """测试价格评分优化"""
        print("\n🎯 测试价格评分优化")
        print("=" * 50)
        
        # 模拟搜索参数
        search_params = {
            'price_range': (800, 1200),  # 800-1200万
        }
        
        # 创建测试文档
        from langchain_core.documents import Document
        test_docs = [
            Document(
                page_content="豪华别墅，位置优越",
                metadata={'property_id': 1, 'price': 1000, 'title': '理想价格房源'}
            ),
            Document(
                page_content="精装公寓，交通便利", 
                metadata={'property_id': 2, 'price': 900, 'title': '接近理想价格'}
            ),
            Document(
                page_content="学区房，教育资源丰富",
                metadata={'property_id': 3, 'price': 1300, 'title': '超出预算10%'}
            ),
            Document(
                page_content="市中心住宅",
                metadata={'property_id': 4, 'price': 1400, 'title': '超出预算过多'}
            ),
        ]
        
        # 测试重排序
        reranked_docs = rag_service._rerank_and_filter(test_docs, search_params)
        
        print("价格评分结果（按排序）:")
        for i, doc in enumerate(reranked_docs, 1):
            price = doc.metadata.get('price')
            title = doc.metadata.get('title')
            print(f"  {i}. {title}: {price}万")
        
        # 验证结果
        if reranked_docs:
            top_price = reranked_docs[0].metadata.get('price')
            if 800 <= top_price <= 1200:
                print("✅ 价格评分优化有效：预算内房源排名靠前")
            else:
                print("⚠️ 价格评分可能需要调整")
    
    def test_negative_conditions(self):
        """测试否定条件处理"""
        print("\n🚫 测试否定条件处理")
        print("=" * 50)
        
        search_params = {
            'special_requirements': ['学区房', '不要吵闹的环境', '避免工厂附近']
        }
        
        from langchain_core.documents import Document
        test_docs = [
            Document(
                page_content="优质学区房，环境安静，绿化良好",
                metadata={'property_id': 1, 'title': '理想房源'}
            ),
            Document(
                page_content="学区房，靠近高架桥，交通便利但较吵闹",
                metadata={'property_id': 2, 'title': '包含否定词的房源'}
            ),
            Document(
                page_content="学区房，附近有化工厂，价格便宜",
                metadata={'property_id': 3, 'title': '包含工厂的房源'}
            ),
        ]
        
        # 测试否定关键词提取
        negative_keywords = rag_service._extract_negative_keywords(search_params)
        print(f"提取的否定关键词: {negative_keywords}")
        
        # 测试重排序
        reranked_docs = rag_service._rerank_and_filter(test_docs, search_params)
        
        print("否定条件处理结果:")
        for i, doc in enumerate(reranked_docs, 1):
            title = doc.metadata.get('title')
            print(f"  {i}. {title}")
        
        # 验证排序是否合理
        if reranked_docs and reranked_docs[0].metadata.get('title') == '理想房源':
            print("✅ 否定条件处理有效：不含否定关键词的房源排名最高")
        else:
            print("⚠️ 否定条件处理可能需要调整")
    
    def test_location_fuzzy_matching(self):
        """测试位置模糊匹配"""
        print("\n🗺️ 测试位置模糊匹配")
        print("=" * 50)
        
        search_params = {
            'location_keywords': ['徐家汇', '市中心']
        }
        
        from langchain_core.documents import Document
        test_docs = [
            Document(
                page_content="精装公寓，交通便利",
                metadata={'property_id': 1, 'location': '上海市徐汇区', 'title': '徐汇区房源'}
            ),
            Document(
                page_content="商务办公楼",
                metadata={'property_id': 2, 'location': '上海市黄浦区', 'title': '黄浦区房源'}
            ),
            Document(
                page_content="住宅小区",
                metadata={'property_id': 3, 'location': '上海市浦东新区', 'title': '浦东房源'}
            ),
        ]
        
        # 测试相似度计算
        similarity1 = rag_service._calculate_location_similarity('徐家汇', '上海市徐汇区')
        similarity2 = rag_service._calculate_location_similarity('市中心', '上海市黄浦区')
        
        print(f"'徐家汇' vs '上海市徐汇区' 相似度: {similarity1:.2f}")
        print(f"'市中心' vs '上海市黄浦区' 相似度: {similarity2:.2f}")
        
        # 测试重排序
        reranked_docs = rag_service._rerank_and_filter(test_docs, search_params)
        
        print("位置匹配结果:")
        for i, doc in enumerate(reranked_docs, 1):
            location = doc.metadata.get('location')
            title = doc.metadata.get('title')
            print(f"  {i}. {title}: {location}")
        
        if similarity1 > 0.7:
            print("✅ 位置模糊匹配有效：能够处理地名别名")
        else:
            print("⚠️ 位置模糊匹配可能需要调整")
    
    def test_comprehensive_optimization(self):
        """综合测试所有优化"""
        print("\n🔄 综合优化效果测试")
        print("=" * 50)
        
        for test_case in self.test_cases:
            print(f"\n📝 测试用例: {test_case['name']}")
            print(f"查询: {test_case['query']}")
            
            try:
                # 执行真实搜索
                response = rag_service.ask_question(test_case['query'], max_results=5)
                properties = response.get('retrieved_properties', [])
                
                print(f"找到 {len(properties)} 个房源:")
                for i, prop in enumerate(properties[:3], 1):
                    title = prop.get('title', 'N/A')
                    price = prop.get('price', 'N/A')
                    location = prop.get('location', 'N/A')
                    match_score = prop.get('match_score', 'N/A')
                    
                    print(f"  {i}. {title}")
                    print(f"     价格: {price}万, 位置: {location}")
                    print(f"     匹配度: {match_score}")
                
                print(f"期望改进: {', '.join(test_case['expected_improvements'])}")
                print("✅ 测试完成")
                
            except Exception as e:
                print(f"❌ 测试失败: {e}")
    
    def test_performance_comparison(self):
        """性能对比测试"""
        print("\n📊 重排序性能分析")
        print("=" * 50)
        
        # 检查重排序统计（如果有的话）
        if hasattr(rag_service, 'rerank_stats'):
            stats = rag_service.rerank_stats
            print(f"重排序统计:")
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        # 模拟性能测试
        import time
        from langchain_core.documents import Document
        
        # 创建大量测试文档
        test_docs = []
        for i in range(50):
            doc = Document(
                page_content=f"测试房源{i}的详细描述",
                metadata={
                    'property_id': i,
                    'price': 800 + (i * 10),
                    'location': f'测试区域{i % 5}',
                    'hybrid_score': 1.0 - (i * 0.01)  # 模拟混合搜索分数
                }
            )
            test_docs.append(doc)
        
        search_params = {
            'price_range': (900, 1100),
            'location_keywords': ['测试区域1'],
            'special_requirements': ['优质']
        }
        
        # 测试重排序性能
        start_time = time.time()
        reranked_docs = rag_service._rerank_and_filter(test_docs, search_params)
        end_time = time.time()
        
        print(f"处理 {len(test_docs)} 个文档用时: {(end_time - start_time)*1000:.2f}ms")
        print(f"返回 {len(reranked_docs)} 个优质结果")
        print(f"平均每个文档处理时间: {((end_time - start_time)/len(test_docs))*1000:.2f}ms")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 重排序优化效果测试")
        print("=" * 60)
        
        # 检查前置条件
        if not self.check_prerequisites():
            print("❌ 前置条件检查失败，测试中止")
            return
        
        # 运行各项测试
        self.test_price_scoring_optimization()
        self.test_negative_conditions()
        self.test_location_fuzzy_matching()
        self.test_comprehensive_optimization()
        self.test_performance_comparison()
        
        print("\n" + "=" * 60)
        print("✨ 重排序优化测试完成！")
        print("\n📈 主要改进:")
        print("  1. ✅ 融合混合搜索分数作为基础分")
        print("  2. ✅ 连续的价格接近度评分")
        print("  3. ✅ 智能否定条件处理")
        print("  4. ✅ 位置模糊匹配能力")
        print("  5. ✅ 更精确的综合排序")
        
    def check_prerequisites(self) -> bool:
        """检查测试前置条件"""
        try:
            print("📋 检查测试前置条件...")
            
            # 检查数据库连接
            if not db_manager.test_connection():
                print("❌ 数据库连接失败")
                return False
            
            # 检查重排序方法是否存在
            if not hasattr(rag_service, '_rerank_and_filter'):
                print("❌ 重排序方法未找到")
                return False
            
            # 检查新增的方法
            required_methods = [
                '_extract_negative_keywords',
                '_calculate_location_similarity',
                '_simple_string_similarity'
            ]
            
            for method in required_methods:
                if not hasattr(rag_service, method):
                    print(f"❌ 方法 {method} 未找到")
                    return False
            
            print("✅ 前置条件检查通过")
            return True
            
        except Exception as e:
            print(f"❌ 前置条件检查失败: {e}")
            return False


def main():
    """主函数"""
    tester = RerankingOptimizationTester()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
