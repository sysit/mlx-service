#!/usr/bin/env python3
"""测试音频转录 API"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
import os

BASE_URL = "http://localhost:11434"
API_KEY = "sk-prod-xxx"


def test_audio_transcription():
    """测试音频转录"""
    # 需要一个测试音频文件
    # 这里只测试 API 端点是否存在
    audio_file = os.path.expanduser("~/test_audio.wav")
    
    if not os.path.exists(audio_file):
        print(f"⚠️  测试音频文件不存在: {audio_file}")
        print("   请提供一个测试音频文件")
        return
    
    with open(audio_file, "rb") as f:
        r = requests.post(
            f"{BASE_URL}/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {API_KEY}"},
            files={"file": f},
            data={"model": "whisper-large-v3-fp16"}
        )
    
    if r.status_code == 200:
        data = r.json()
        print(f"✅ 音频转录: {data.get('text', '')[:100]}...")
    else:
        print(f"❌ 音频转录失败: {r.status_code} {r.text}")


def test_models_include_audio():
    """测试模型列表包含音频模型"""
    r = requests.get(f"{BASE_URL}/v1/models", headers={"Authorization": f"Bearer {API_KEY}"})
    if r.status_code == 200:
        data = r.json()
        audio_models = [m for m in data.get("data", []) if "whisper" in m.get("id", "").lower()]
        if audio_models:
            print(f"✅ 发现音频模型: {[m['id'] for m in audio_models]}")
        else:
            print("⚠️  未发现音频模型")
    else:
        print(f"❌ 获取模型列表失败: {r.status_code}")


if __name__ == "__main__":
    try:
        test_models_include_audio()
        test_audio_transcription()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")