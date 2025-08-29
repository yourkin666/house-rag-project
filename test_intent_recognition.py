#!/usr/bin/env python3
"""
测试优化后的意图识别效果
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from house_rag.core.embeddings import HouseRAGEmbeddings
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)

def test_intent_recognition():
    """测试不同查询的意图识别效果和成本优化"""
    
    # 创建RAG实例（需要确保有API key）
    try:
        rag = HouseRAGEmbeddings()
    except Exception as e:
        print(f"无法初始化RAG系统: {e}")
        print("请确保GOOGLE_API_KEY环境变量已设置")
        return
    
    # 测试查询样例（包含不同复杂度）
    test_queries = [
        # 简单查询（应该使用关键词匹配）
        ("学区房", "简单查询"),
        ("推荐房子", "探索性查询"),
        ("有什么房源", "模糊查询"),
        
        # 复杂查询（应该使用LLM）
        ("我想找便宜一点的房子，性价比要高", "价格敏感复合"),
        ("有没有陆家嘴的顶级豪华别墅？", "高端+位置复合"),
        ("不要太偏远的学区房", "否定词+特殊需求"),
        ("性价比高的地铁沿线房源", "价格+特殊需求复合"),
        ("不考虑老破小，要有电梯的", "否定词+特殊需求"),
        
        # 重复查询（测试缓存）
        ("学区房", "重复简单查询"),
        ("不要太偏远的学区房", "重复复杂查询"),
    ]
    
    print("=== 成本优化的意图识别测试 ===\n")
    
    for i, (query, query_type) in enumerate(test_queries, 1):
        print(f"测试 {i} ({query_type}): \"{query}\"")
        try:
            # 测试完整的配置生成过程
            dynamic_k = 5  # 示例动态K值
            config = rag._get_adaptive_retriever_config(query, dynamic_k)
            print(f"检索配置: {config}")
            
            # 显示是否使用了LLM
            should_use_llm = rag._should_use_llm_analysis(query) if not hasattr(rag, '_intent_cache') or f"intent_{hash(query)}" not in rag._intent_cache else False
            print(f"是否使用LLM: {'是' if should_use_llm else '否（关键词匹配或缓存）'}")
            
        except Exception as e:
            print(f"测试失败: {e}")
        
        print("-" * 50)
    
    # 显示成本统计
    print("\n=== 成本统计 ===")
    stats = rag.get_cost_stats()
    print(f"总查询次数: {len(test_queries)}")
    print(f"LLM调用次数: {stats['total_calls']}")
    print(f"缓存命中次数: {stats['cache_hits']}")
    print(f"关键词回退次数: {stats['keyword_fallbacks']}")
    print(f"缓存命中率: {stats['cache_hit_rate']:.1f}%")
    print(f"LLM使用率: {stats['total_calls']/len(test_queries)*100:.1f}%")
    
    # 估算成本节省
    potential_cost_without_optimization = len(test_queries)
    actual_cost = stats['total_calls']
    cost_savings_percent = (potential_cost_without_optimization - actual_cost) / potential_cost_without_optimization * 100
    print(f"成本节省: {cost_savings_percent:.1f}% (从{potential_cost_without_optimization}次降至{actual_cost}次LLM调用)")

if __name__ == "__main__":
    test_intent_recognition()
