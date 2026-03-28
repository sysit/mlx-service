#!/usr/bin/env python3
"""全量回归测试"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import json

BASE_URL = "http://localhost:11434"
API_KEY = "sk-prod-xxx"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


def test_health():
    """健康检查"""
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"健康检查失败: {r.status_code}"
    print("✅ 健康检查")


def test_models_list():
    """模型列表"""
    r = requests.get(f"{BASE_URL}/v1/models", headers=HEADERS)
    assert r.status_code == 200, f"模型列表失败: {r.status_code}"
    data = r.json()
    assert "data" in data
    models = [m["id"] for m in data["data"]]
    print(f"✅ 模型列表: {models}")
    return models


def test_models_loaded():
    """已加载模型"""
    r = requests.get(f"{BASE_URL}/v1/models/loaded", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    print(f"✅ 已加载模型: {len(data['models'])} 个")


def test_chat_completions_sync():
    """同步聊天补全"""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "qwen3-vl-8b",
            "messages": [{"role": "user", "content": "Say 'test ok'"}],
            "max_tokens": 20
        }
    )
    assert r.status_code == 200, f"同步聊天失败: {r.text}"
    data = r.json()
    assert "choices" in data
    content = data["choices"][0]["message"]["content"]
    print(f"✅ 同步聊天: {content[:30]}...")


def test_chat_completions_stream():
    """流式聊天补全"""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "qwen3-vl-8b",
            "messages": [{"role": "user", "content": "Count 1 to 3"}],
            "max_tokens": 20,
            "stream": True
        },
        stream=True
    )
    assert r.status_code == 200, f"流式聊天失败: {r.status_code}"
    
    chunks = []
    for line in r.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data != "[DONE]":
                    chunk = json.loads(data)
                    if chunk["choices"][0]["delta"].get("content"):
                        chunks.append(chunk["choices"][0]["delta"]["content"])
    
    content = "".join(chunks)
    print(f"✅ 流式聊天: {content[:30]}...")


def test_vl_image():
    """VL 图片识别"""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "qwen3-vl-8b",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this? One word."},
                    {"type": "image_url", "image_url": {"url": "/tmp/test_image.jpg"}}
                ]
            }],
            "max_tokens": 20
        }
    )
    assert r.status_code == 200, f"VL 图片识别失败: {r.text}"
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print(f"✅ VL 图片识别: {content[:30]}...")


def test_vl_text_only():
    """VL 纯文本（无图片）"""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "qwen3-vl-8b",
            "messages": [{"role": "user", "content": "Say 'vl ok'"}],
            "max_tokens": 20
        }
    )
    assert r.status_code == 200, f"VL 纯文本失败: {r.text}"
    data = r.json()
    content = data["choices"][0]["message"]["content"]
    print(f"✅ VL 纯文本: {content[:30]}...")


def test_ollama_tags():
    """Ollama API - 模型标签"""
    r = requests.get(f"{BASE_URL}/api/tags", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    print(f"✅ Ollama /api/tags: {len(data.get('models', []))} models")


def test_ollama_generate():
    """Ollama API - 生成"""
    r = requests.post(
        f"{BASE_URL}/api/generate",
        headers=HEADERS,
        json={
            "model": "qwen3-vl-8b",
            "prompt": "Say 'ollama ok'",
            "stream": False
        }
    )
    assert r.status_code == 200, f"Ollama generate 失败: {r.text}"
    data = r.json()
    print(f"✅ Ollama /api/generate: {data.get('response', '')[:30]}...")


def test_cache_stats():
    """缓存统计"""
    r = requests.get(f"{BASE_URL}/v1/cache/stats", headers=HEADERS)
    assert r.status_code == 200
    data = r.json()
    print(f"✅ 缓存统计: enabled={data.get('enabled', False)}")


def test_auth_invalid():
    """无效认证"""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json={
            "model": "qwen3-vl-8b",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10
        }
    )
    assert r.status_code == 401, f"无效认证应返回 401，实际: {r.status_code}"
    print("✅ 无效认证拒绝")


def test_model_not_found():
    """模型不存在"""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        headers=HEADERS,
        json={
            "model": "nonexistent-model",
            "messages": [{"role": "user", "content": "test"}],
            "max_tokens": 10
        }
    )
    assert r.status_code in [400, 404], f"模型不存在应返回 400/404，实际: {r.status_code}"
    print("✅ 模型不存在处理")


def run_all():
    """运行所有测试"""
    tests = [
        ("健康检查", test_health),
        ("模型列表", test_models_list),
        ("已加载模型", test_models_loaded),
        ("同步聊天", test_chat_completions_sync),
        ("流式聊天", test_chat_completions_stream),
        ("VL 图片识别", test_vl_image),
        ("VL 纯文本", test_vl_text_only),
        ("Ollama Tags", test_ollama_tags),
        ("Ollama Generate", test_ollama_generate),
        ("缓存统计", test_cache_stats),
        ("无效认证", test_auth_invalid),
        ("模型不存在", test_model_not_found),
    ]
    
    passed = 0
    failed = 0
    
    print("\n" + "=" * 50)
    print("MLX Service 全量回归测试")
    print("=" * 50 + "\n")
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"❌ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {name}: 异常 - {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"测试结果: ✅ {passed} 通过, ❌ {failed} 失败")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)