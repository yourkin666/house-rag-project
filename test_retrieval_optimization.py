#!/usr/bin/env python3
"""
检索策略优化测试脚本
用于验证和演示优化后的RAG检索效果
"""

import sys
import os
import json
from typing import List, Dict

# 添加项目路径
sys.path.append('/Users/apple/Desktop/house-rag-project/src')

from house_rag.core.embeddings import rag_service

def test_query_analysis():
    """测试查询参数提取功能"""
    print("=== 测试查询参数提取 ===")
    
    test_queries = [
        "我想在浦东新区找一套200-300万的房子",
        "有没有带地铁站附近的学区房",
        "预算150万左右，最好有停车位",
        "找个100平方左右的房子，要朝南",
        "虹口区或者黄浦区的公寓，不超过250万"
    ]
    
    for query in test_queries:
        params = rag_service._extract_search_parameters(query)
        print(f"\n查询: {query}")
        print(f"提取参数: {json.dumps(params, ensure_ascii=False, indent=2)}")

def test_dynamic_k_calculation():
    """测试动态K值计算"""
    print("\n=== 测试动态K值计算 ===")
    
    test_queries = [
        "找房子",  # 简单查询
        "我想在上海找一套房子，价格合理",  # 中等复杂度
        "我需要在浦东新区找一套200-300万的三房两厅，最好靠近地铁站，还要有学区，并且有停车位"  # 复杂查询
    ]
    
    for query in test_queries:
        k_value = rag_service._calculate_dynamic_k(query)
        print(f"查询: {query}")
        print(f"动态K值: {k_value}, 查询长度: {len(query)}")

def test_retrieval_quality():
    """测试检索质量"""
    print("\n=== 测试检索质量 ===")
    
    test_queries = [
        "浦东新区200万左右的房子",
        "有地铁的学区房",
        "便宜的房子"
    ]
    
    for query in test_queries:
        print(f"\n查询: {query}")
        try:
            result = rag_service.query_properties(query, max_results=3)
            
            print(f"搜索质量: {result['search_quality']}")
            print(f"查询分析: {result['query_analysis']}")
            
            print("匹配房源:")
            for i, prop in enumerate(result['retrieved_properties'], 1):
                print(f"  {i}. {prop['title']} - {prop['price']}万")
                print(f"     匹配度: {prop['match_percentage']}%")
                print(f"     匹配原因: {prop['match_reasons']}")
                
        except Exception as e:
            print(f"测试失败: {e}")

def test_caching():
    """测试缓存机制"""
    print("\n=== 测试缓存机制 ===")
    
    query = "浦东新区的房子"
    
    # 第一次查询
    print("第一次查询...")
    import time
    start_time = time.time()
    result1 = rag_service.query_properties(query)
    time1 = time.time() - start_time
    
    # 第二次查询（应该使用缓存）
    print("第二次查询...")
    start_time = time.time()
    result2 = rag_service.query_properties(query)
    time2 = time.time() - start_time
    
    print(f"第一次查询时间: {time1:.3f}秒")
    print(f"第二次查询时间: {time2:.3f}秒")
    print(f"第二次是否使用缓存: {result2['search_quality']['used_cache']}")
    print(f"缓存大小: {len(rag_service._query_cache)}")

def performance_comparison():
    """性能对比测试"""
    print("\n=== 性能对比测试 ===")
    
    queries = [
        "上海200万的房子",
        "浦东新区带地铁的房源",
        "学区房推荐"
    ]
    
    import time
    
    for query in queries:
        print(f"\n测试查询: {query}")
        
        # 测试多次查询的性能
        times = []
        for i in range(3):
            start_time = time.time()
            result = rag_service.query_properties(query)
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if i == 0:
                print(f"  搜索质量等级: {result['search_quality']['search_quality_level']}")
                print(f"  平均匹配分数: {result['search_quality']['average_match_score']}")
        
        avg_time = sum(times) / len(times)
        print(f"  平均响应时间: {avg_time:.3f}秒")

def main():
    """主测试函数"""
    print("开始检索策略优化测试...")
    
    try:
        # 测试各个功能模块
        test_query_analysis()
        test_dynamic_k_calculation()
        test_retrieval_quality()
        test_caching()
        performance_comparison()
        
        print("\n=== 测试完成 ===")
        print("所有检索优化功能测试成功！")
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
