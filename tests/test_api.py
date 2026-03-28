#!/usr/bin/env python3
"""测试 API 端点"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

BASE_URL = "http://localhost:11434"
API_KEY = "sk-prod-xxx"


def test_health():
    """健康检查"""
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    print("✅ 健康检查")


def test_models():
    """模型列表"""
    r = requests.get(f"{BASE_URL}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 200
    data = r.json()
    assert "data" in data
    print(f"✅ 模型列表 ({len(data['data'])} models)")


def test_chat_completions():
    """聊天补全"""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
        json={"model": "qwen3.5-122b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 10}
    )
    assert r.status_code == 200
    data = r.json()
    assert "choices" in data
    print(f"✅ 聊天补全: {data['choices'][0]['message']['content'][:20]}...")


def test_ollama_api():
    """Ollama API"""
    r = requests.get(f"{BASE_URL}/api/tags", headers={"Authorization": f"Bearer {API_KEY}"})
    assert r.status_code == 200
    print("✅ Ollama /api/tags")


if __name__ == "__main__":
    try:
        test_health()
        test_models()
        test_chat_completions()
        test_ollama_api()
        print("\n✅ 所有测试通过")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")