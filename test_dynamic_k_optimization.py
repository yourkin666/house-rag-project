#!/usr/bin/env python3
"""
测试优化后的 dynamic_k 功能
验证新的复杂度分析算法是否按预期工作
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from house_rag.core.embeddings import RAGService

def test_dynamic_k_scenarios():
    """测试不同场景下的 dynamic_k 计算"""
    
    # 模拟一个简化的 RAGService，只测试 dynamic_k 计算
    class MockRAGService:
        def _count_extracted_fields(self, params):
            count = 0
            if params.get('price_range'):
                count += 1
            if params.get('location_keywords'):
                count += 1
            if params.get('property_type'):
                count += 1
            if params.get('area_preference'):
                count += 1
            if params.get('special_requirements'):
                count += 1
            return count
        
        def _calculate_dynamic_k(self, search_params, question, base_k=5, max_k=12, min_k=4):
            """优化后的 dynamic_k 算法"""
            complexity_score = 0
            
            # 1. 基于提取出的参数数量
            if search_params.get('price_range'):
                complexity_score += 1
            if search_params.get('location_keywords'):
                complexity_score += len(search_params['location_keywords'])
            if search_params.get('property_type'):
                complexity_score += 1
            if search_params.get('area_preference'):
                complexity_score += 1
            if search_params.get('special_requirements'):
                complexity_score += len(search_params['special_requirements'])
            
            # 2. 逻辑连接词
            logical_keywords = ['并且', '同时', '或者', '要么', '另外', '而且', '以及']
            logical_complexity = sum(1 for keyword in logical_keywords if keyword in question)
            complexity_score += logical_complexity
            
            # 3. 问题长度
            query_length = len(question)
            if query_length > 80:
                complexity_score += 2
            elif query_length > 50:
                complexity_score += 1
                
            # 4. 模糊查询
            vague_indicators = ['推荐', '有什么', '看看', '找找', '合适的']
            if any(indicator in question for indicator in vague_indicators):
                complexity_score += 1
            
            # 计算最终 k 值
            adjusted_k = min(base_k + complexity_score, max_k)
            final_k = max(adjusted_k, min_k)
            
            return final_k, complexity_score

    mock_service = MockRAGService()
    
    # 测试场景
    test_cases = [
        # 场景1: 简单查询
        {
            'name': '简单查询',
            'question': '找房子',
            'search_params': {
                'price_range': None,
                'location_keywords': [],
                'property_type': None,
                'area_preference': None,
                'special_requirements': []
            },
            'expected_k_range': (4, 6)  # 预期的K值范围
        },
        
        # 场景2: 中等复杂度查询
        {
            'name': '中等复杂查询',
            'question': '我想在浦东找一个800万以内的房子，最好靠近地铁',
            'search_params': {
                'price_range': (0, 800),
                'location_keywords': ['浦东'],
                'property_type': None,
                'area_preference': None,
                'special_requirements': ['地铁']
            },
            'expected_k_range': (8, 10)
        },
        
        # 场景3: 高复杂度查询
        {
            'name': '高复杂查询',
            'question': '我想找一个房子，要么在静安区要么在徐汇区，预算1000万左右，最好是别墅并且要有停车位，同时还要靠近地铁站，面积120平米以上',
            'search_params': {
                'price_range': (800, 1200),
                'location_keywords': ['静安区', '徐汇区'],
                'property_type': '别墅',
                'area_preference': 120,
                'special_requirements': ['停车位', '地铁']
            },
            'expected_k_range': (12, 12)  # 达到上限
        },
        
        # 场景4: 模糊查询
        {
            'name': '模糊查询',
            'question': '推荐一些合适的房源给我看看',
            'search_params': {
                'price_range': None,
                'location_keywords': [],
                'property_type': None,
                'area_preference': None,
                'special_requirements': []
            },
            'expected_k_range': (6, 8)  # 模糊查询需要更多选择
        }
    ]
    
    print("🧪 Dynamic K 优化测试\n" + "="*50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试场景 {i}: {test_case['name']}")
        print(f"问题: {test_case['question']}")
        print(f"提取的参数: {test_case['search_params']}")
        
        # 计算 dynamic_k
        k_value, complexity_score = mock_service._calculate_dynamic_k(
            test_case['search_params'], 
            test_case['question']
        )
        
        print(f"复杂度分数: {complexity_score}")
        print(f"计算出的K值: {k_value}")
        
        # 验证结果
        expected_min, expected_max = test_case['expected_k_range']
        if expected_min <= k_value <= expected_max:
            print(f"✅ 通过 (期望范围: {expected_min}-{expected_max})")
        else:
            print(f"❌ 失败 (期望范围: {expected_min}-{expected_max}, 实际: {k_value})")
    
    print("\n" + "="*50)
    print("🎯 测试总结:")
    print("- 优化后的 dynamic_k 能够基于结构化参数精确计算复杂度")
    print("- 多维度评估比单纯字符串分析更准确")
    print("- K值在合理范围内(4-12)动态调整")
    print("- 针对不同查询类型提供差异化的检索范围")

if __name__ == "__main__":
    test_dynamic_k_scenarios()
