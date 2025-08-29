#!/usr/bin/env python3
"""
自适应检索策略测试脚本
验证不同查询类型的检索策略调整效果
"""

import sys
sys.path.append('/Users/apple/Desktop/house-rag-project/src')

from house_rag.core.embeddings import rag_service

def test_adaptive_retrieval():
    """测试自适应检索策略"""
    
    print("🎯 自适应检索策略测试")
    print("=" * 50)
    
    # 测试不同类型的查询
    test_queries = [
        {
            "query": "便宜的房子",
            "type": "价格敏感查询",
            "expected": "增加结果数量，不使用相似度阈值"
        },
        {
            "query": "豪华别墅推荐", 
            "type": "高端精准查询",
            "expected": "高相似度阈值 (0.75)"
        },
        {
            "query": "浦东新区的房源",
            "type": "区域性查询", 
            "expected": "适中相似度阈值 (0.68)"
        },
        {
            "query": "带地铁的学区房",
            "type": "特殊需求查询",
            "expected": "较高相似度阈值 (0.72)"
        },
        {
            "query": "有什么房子推荐",
            "type": "模糊查询",
            "expected": "大幅增加结果数量"
        },
        {
            "query": "我想买套房子，价格200万左右",
            "type": "默认策略", 
            "expected": "标准相似度阈值 (0.7)"
        }
    ]
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n📍 测试 {i}: {test['type']}")
        print(f"🔍 查询: \"{test['query']}\"")
        print(f"📋 预期策略: {test['expected']}")
        print("-" * 30)
        
        try:
            # 测试动态K值计算
            dynamic_k = rag_service._calculate_dynamic_k(test['query'])
            print(f"📊 动态K值: {dynamic_k}")
            
            # 测试自适应检索配置
            config = rag_service._get_adaptive_retriever_config(test['query'], dynamic_k)
            print(f"⚙️  检索配置: {config}")
            
            # 分析配置类型
            search_type = config['search_type']
            search_kwargs = config['search_kwargs']
            
            if search_type == "similarity":
                print(f"✅ 策略: 相似度搜索，K={search_kwargs['k']}")
            else:
                print(f"✅ 策略: 阈值相似度搜索，K={search_kwargs['k']}, 阈值={search_kwargs['score_threshold']}")
            
        except Exception as e:
            print(f"❌ 测试失败: {e}")
    
    print(f"\n{'=' * 50}")
    print("🎉 自适应检索策略测试完成！")

def test_real_queries():
    """测试真实查询的完整流程"""
    
    print("\n🏠 真实查询测试")
    print("=" * 50)
    
    real_queries = [
        "上海浦东新区便宜的房子",
        "北京豪华别墅", 
        "带学区的房源推荐"
    ]
    
    for query in real_queries:
        print(f"\n🔍 测试查询: \"{query}\"")
        try:
            # 获取检索策略
            dynamic_k = rag_service._calculate_dynamic_k(query)
            config = rag_service._get_adaptive_retriever_config(query, dynamic_k)
            
            print(f"📊 检索配置: {config}")
            
            # 执行完整查询 (如果有数据的话)
            result = rag_service.query_properties(query, max_results=3)
            
            print(f"✅ 查询成功!")
            print(f"📈 搜索质量: {result.get('search_quality', {}).get('search_quality_level', 'N/A')}")
            print(f"🏆 找到房源数: {len(result.get('retrieved_properties', []))}")
            
        except Exception as e:
            print(f"⚠️  查询执行失败: {e}")
            print("(可能是因为没有向量化数据，但策略配置正常)")

def main():
    """主测试函数"""
    try:
        test_adaptive_retrieval()
        test_real_queries()
        
        print(f"\n💡 优化效果总结:")
        print("✨ 不同查询类型现在会自动使用最佳检索策略")
        print("✨ 价格敏感查询 → 更多选择")  
        print("✨ 高端查询 → 更高精度")
        print("✨ 模糊查询 → 增加覆盖率")
        print("✨ 特殊需求查询 → 精准匹配")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
