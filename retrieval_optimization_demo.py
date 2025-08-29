#!/usr/bin/env python3
"""
检索策略优化演示脚本
展示优化后的RAG系统如何智能处理房源查询
"""

import sys
import json
from typing import Dict

# 添加项目路径
sys.path.append('/Users/apple/Desktop/house-rag-project/src')

from house_rag.core.embeddings import rag_service

def demo_intelligent_search():
    """演示智能搜索功能"""
    print("🏠 房源智能检索系统演示")
    print("=" * 50)
    
    # 模拟不同类型的用户查询
    demo_queries = [
        {
            "query": "我想在浦东新区找一套200-300万的房子，最好靠近地铁",
            "description": "复杂条件查询 - 包含价格范围、地理位置和特殊需求"
        },
        {
            "query": "有没有100平方左右的学区房",
            "description": "特殊需求查询 - 包含面积和特殊要求"
        },
        {
            "query": "便宜的公寓",
            "description": "模糊查询 - 测试系统的理解能力"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n📍 演示 {i}: {demo['description']}")
        print(f"🔍 用户问题: \"{demo['query']}\"")
        print("-" * 40)
        
        try:
            # 使用优化后的查询方法
            result = rag_service.query_properties(demo['query'], max_results=3)
            
            # 显示查询分析
            print("📊 查询分析:")
            analysis = result['query_analysis']
            if analysis.get('price_range'):
                print(f"  💰 价格范围: {analysis['price_range'][0]}-{analysis['price_range'][1]}万")
            if analysis.get('location_keywords'):
                print(f"  📍 位置关键词: {', '.join(analysis['location_keywords'])}")
            if analysis.get('special_requirements'):
                print(f"  ⭐ 特殊需求: {', '.join(analysis['special_requirements'])}")
            if analysis.get('property_type'):
                print(f"  🏡 房屋类型: {analysis['property_type']}")
            
            # 显示搜索质量
            quality = result['search_quality']
            print(f"\n📈 搜索质量:")
            print(f"  🎯 质量等级: {quality['search_quality_level']}")
            print(f"  📊 平均匹配分数: {quality['average_match_score']}")
            print(f"  🔢 找到房源数: {quality['total_found']} → 返回: {quality['returned_count']}")
            print(f"  ⚡ 使用缓存: {'是' if quality['used_cache'] else '否'}")
            
            # 显示匹配的房源
            print(f"\n🏆 推荐房源:")
            for j, prop in enumerate(result['retrieved_properties'], 1):
                print(f"  {j}. {prop['title']}")
                print(f"     📍 {prop['location']} | 💰 {prop['price']}万")
                print(f"     🎯 匹配度: {prop['match_percentage']}%")
                if prop['match_reasons']:
                    print(f"     ✅ 匹配原因: {', '.join(prop['match_reasons'])}")
            
            # 显示AI回答片段
            answer_preview = result['answer'][:200] + "..." if len(result['answer']) > 200 else result['answer']
            print(f"\n🤖 AI回答预览:")
            print(f"  {answer_preview}")
            
        except Exception as e:
            print(f"❌ 演示失败: {e}")
        
        print("\n" + "="*50)

def demo_caching_performance():
    """演示缓存性能提升"""
    print("\n⚡ 缓存性能演示")
    print("=" * 30)
    
    query = "浦东新区200万的房子"
    
    import time
    
    # 第一次查询
    print("🔍 第一次查询 (无缓存)...")
    start_time = time.time()
    result1 = rag_service.query_properties(query)
    time1 = time.time() - start_time
    
    # 第二次查询
    print("🔍 第二次查询 (使用缓存)...")
    start_time = time.time()
    result2 = rag_service.query_properties(query)
    time2 = time.time() - start_time
    
    print(f"\n📊 性能对比:")
    print(f"  第一次查询: {time1:.3f}秒")
    print(f"  第二次查询: {time2:.3f}秒")
    print(f"  性能提升: {((time1 - time2) / time1 * 100):.1f}%")
    print(f"  缓存状态: {'✅ 已使用' if result2['search_quality']['used_cache'] else '❌ 未使用'}")

def show_optimization_features():
    """展示优化功能特性"""
    print("\n🚀 检索优化功能特性")
    print("=" * 30)
    
    features = [
        "🧠 智能查询参数提取 - 自动识别价格、位置、房型等条件",
        "📊 动态检索数量调整 - 根据查询复杂度调整返回结果数",
        "🎯 多维度匹配评分 - 价格、位置、类型、特殊需求综合评分",
        "🔄 结果重排序和过滤 - 基于用户需求对结果进行智能排序",
        "⚡ 查询缓存机制 - 提高重复查询的响应速度",
        "📈 搜索质量分析 - 提供详细的搜索质量评估",
        "🛡️ 降级处理策略 - 确保系统稳定性和可用性"
    ]
    
    for feature in features:
        print(f"  {feature}")

def main():
    """主演示函数"""
    try:
        show_optimization_features()
        demo_intelligent_search()
        demo_caching_performance()
        
        print("\n✅ 检索策略优化演示完成！")
        print("\n💡 优化效果总结:")
        print("  - 更智能的查询理解和参数提取")
        print("  - 更精准的结果匹配和排序")
        print("  - 更快的响应速度（缓存机制）")
        print("  - 更详细的搜索分析和质量评估")
        print("  - 更稳定的系统性能（降级策略）")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
