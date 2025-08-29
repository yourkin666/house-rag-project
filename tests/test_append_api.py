#!/usr/bin/env python3
"""
测试 /append API 接口的脚本

此脚本用于测试新添加的 /append 接口，验证房源数据添加和自动向量化功能。

使用方法：
1. 确保 API 服务正在运行（localhost:8000）
2. 运行此脚本：python test_append_api.py
"""

import requests
import json
import time

# API 基础 URL
BASE_URL = "http://localhost:8000"

def test_health():
    """测试健康检查接口"""
    print("🔍 测试健康检查接口...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过: {data['message']}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"❌ 无法连接到 API 服务: {e}")
        return False

def test_append_api():
    """测试添加房源接口"""
    print("🏠 测试房源添加接口...")
    
    # 测试房源数据
    test_property = {
        "title": "测试豪华海景别墅",
        "location": "广东省珠海市香洲区",
        "price": 1280.0,
        "description": "位于珠海市香洲区海滨的豪华别墅，占地面积500平方米，建筑面积380平方米。房屋面朝大海，拥有私人海滩和花园。内部装修奢华，配备智能家居系统、地暖、中央空调等现代化设施。别墅有6个卧室、4个卫生间、2个客厅和1个书房。小区环境优美，24小时安保，配套设施包括会所、游泳池、网球场等。距离珠海市中心15分钟车程，是度假居住的理想选择。"
    }
    
    try:
        print("📤 正在发送添加房源请求...")
        response = requests.post(
            f"{BASE_URL}/append", 
            headers={"Content-Type": "application/json"},
            json=test_property
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 房源添加成功!")
            print(f"   房源ID: {data.get('property_id')}")
            print(f"   向量化状态: {'已完成' if data.get('embedding_generated') else '未完成'}")
            print(f"   响应消息: {data.get('message')}")
            return data.get('property_id')
        else:
            print(f"❌ 房源添加失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return None
            
    except requests.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return None

def test_ask_with_new_property():
    """使用新添加的房源测试查询接口"""
    print("🤔 测试基于新房源的查询...")
    
    test_question = {
        "question": "珠海有什么豪华海景别墅吗？",
        "max_results": 2
    }
    
    try:
        print("📤 正在发送查询请求...")
        response = requests.post(
            f"{BASE_URL}/ask",
            headers={"Content-Type": "application/json"},
            json=test_question
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ 查询成功!")
            print(f"   AI回答: {data.get('answer')}")
            print(f"   检索到的房源数量: {len(data.get('retrieved_properties', []))}")
            
            # 显示检索到的房源信息
            for i, prop in enumerate(data.get('retrieved_properties', []), 1):
                metadata = prop.get('metadata', {})
                print(f"   房源 {i}: {metadata.get('title', 'N/A')} - {metadata.get('location', 'N/A')}")
            
            return True
        else:
            print(f"❌ 查询失败: {response.status_code}")
            print(f"   错误信息: {response.text}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ 查询请求失败: {e}")
        return False

def get_properties_count():
    """获取房源统计信息"""
    print("📊 获取房源统计信息...")
    
    try:
        response = requests.get(f"{BASE_URL}/properties/count")
        if response.status_code == 200:
            data = response.json()
            print("✅ 统计信息获取成功!")
            print(f"   总房源数: {data.get('total_properties')}")
            print(f"   已向量化: {data.get('embedded_properties')}")
            print(f"   待处理: {data.get('pending_embedding')}")
            return True
        else:
            print(f"❌ 获取统计信息失败: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"❌ 请求失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 开始 API 接口测试")
    print("=" * 50)
    
    # 1. 健康检查
    if not test_health():
        print("💥 健康检查失败，请确保 API 服务正在运行")
        return
    
    print()
    
    # 2. 获取初始统计信息
    print("【测试前统计】")
    get_properties_count()
    print()
    
    # 3. 测试添加房源
    property_id = test_append_api()
    if not property_id:
        print("💥 房源添加测试失败")
        return
    
    print()
    
    # 等待一下确保向量化完成
    print("⏳ 等待2秒确保数据处理完成...")
    time.sleep(2)
    
    # 4. 获取测试后统计信息
    print("【测试后统计】")
    get_properties_count()
    print()
    
    # 5. 测试基于新房源的查询
    if test_ask_with_new_property():
        print("🎉 所有测试通过!")
    else:
        print("⚠️ 查询测试失败，但房源添加成功")
    
    print("=" * 50)
    print("✅ API 测试完成")

if __name__ == "__main__":
    main()
