#!/usr/bin/env python3
"""
MLX Service 性能与边界测试脚本
"""
import asyncio
import json
import time
import os
import psutil
from datetime import datetime
from typing import Dict, List, Any
import subprocess
import sys

# 配置
BASE_URL = "http://localhost:11434"
API_KEY = "sk-xxx"  # 从 start.sh 中的 API_KEYS 配置
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

# 测试报告
report = {
    "test_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "performance": {},
    "boundary": {},
    "stress": {},
    "issues": [],
    "recommendations": []
}


def get_memory_usage() -> float:
    """获取当前进程内存使用(MB)"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def get_system_memory() -> Dict[str, float]:
    """获取系统内存状态"""
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / 1024 / 1024 / 1024,
        "used_gb": mem.used / 1024 / 1024 / 1024,
        "available_gb": mem.available / 1024 / 1024 / 1024,
        "percent": mem.percent
    }


async def call_api(endpoint: str, payload: Dict, timeout: int = 300) -> Dict:
    """调用API"""
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{BASE_URL}{endpoint}",
                json=payload,
                headers=HEADERS,
                timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                return {"status": resp.status, "data": await resp.json()}
    except asyncio.TimeoutError:
        return {"status": 0, "error": "timeout", "error_type": "timeout"}
    except Exception as e:
        return {"status": 0, "error": str(e), "error_type": type(e).__name__}


async def test_model_load_time(model_name: str) -> Dict:
    """测试模型加载时间"""
    print(f"\n📊 测试模型加载时间: {model_name}")
    
    # 清理可能已加载的模型
    unload_payload = {"model": model_name}
    await call_api("/v1/models/unload", unload_payload, timeout=30)
    await asyncio.sleep(2)
    
    mem_before = get_memory_usage()
    start_time = time.time()
    
    # 触发模型加载
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
    }
    
    result = await call_api("/v1/chat/completions", payload, timeout=600)
    
    load_time = time.time() - start_time
    mem_after = get_memory_usage()
    mem_delta = mem_after - mem_before
    
    print(f"  ✅ 加载时间: {load_time:.1f}s, 内存增量: {mem_delta:.1f}MB")
    
    return {
        "model": model_name,
        "load_time_sec": round(load_time, 1),
        "memory_delta_mb": round(mem_delta, 1),
        "success": result.get("status") == 200
    }


async def test_generation_speed(model_name: str, prompt: str = "Write a short story about a robot.", max_tokens: int = 200) -> Dict:
    """测试生成速度"""
    print(f"\n📊 测试生成速度: {model_name}")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False
    }
    
    start_time = time.time()
    result = await call_api("/v1/chat/completions", payload, timeout=300)
    gen_time = time.time() - start_time
    
    tokens_generated = 0
    if result.get("status") == 200:
        content = result["data"].get("choices", [{}])[0].get("message", {}).get("content", "")
        tokens_generated = len(content.split())
    
    tokens_per_sec = tokens_generated / gen_time if gen_time > 0 else 0
    
    print(f"  ✅ 生成时间: {gen_time:.1f}s, tokens: {tokens_generated}, 速度: {tokens_per_sec:.1f} tokens/s")
    
    return {
        "model": model_name,
        "generation_time_sec": round(gen_time, 2),
        "tokens_generated": tokens_generated,
        "tokens_per_sec": round(tokens_per_sec, 2),
        "success": result.get("status") == 200
    }


async def test_concurrent_requests(model_name: str, num_requests: int = 3) -> Dict:
    """测试并发请求"""
    print(f"\n📊 测试并发请求: {num_requests} 个请求同时发送")
    
    # 先加载模型
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 1
    }
    await call_api("/v1/chat/completions", payload, timeout=300)
    await asyncio.sleep(1)
    
    # 发起并发请求
    start_time = time.time()
    
    tasks = []
    for i in range(num_requests):
        task_payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": f"Count to {i+3}"}],
            "max_tokens": 20
        }
        tasks.append(call_api("/v1/chat/completions", task_payload, timeout=120))
    
    results = await asyncio.gather(*tasks)
    
    total_time = time.time() - start_time
    success_count = sum(1 for r in results if r.get("status") == 200)
    
    print(f"  ✅ 并发 {num_requests} 请求完成: 总耗时 {total_time:.1f}s, 成功 {success_count}/{num_requests}")
    
    return {
        "num_requests": num_requests,
        "total_time_sec": round(total_time, 2),
        "success_count": success_count,
        "success": success_count == num_requests
    }


async def test_long_input(model_name: str) -> Dict:
    """测试超长输入"""
    print(f"\n📊 测试超长输入")
    
    # 创建 10000 字符的输入
    long_prompt = "Hello. " * 2000  # 约 13000 字符
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": long_prompt}],
        "max_tokens": 50
    }
    
    result = await call_api("/v1/chat/completions", payload, timeout=300)
    
    success = result.get("status") == 200
    print(f"  ✅ 超长输入处理: {'成功' if success else '失败'} (状态码: {result.get('status')})")
    
    return {
        "input_length": len(long_prompt),
        "status": result.get("status"),
        "success": success,
        "error": result.get("error", "")
    }


async def test_empty_input(model_name: str) -> Dict:
    """测试空输入"""
    print(f"\n📊 测试空输入")
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": ""}],
        "max_tokens": 50
    }
    
    result = await call_api("/v1/chat/completions", payload, timeout=60)
    
    success = result.get("status") == 200
    print(f"  ✅ 空输入处理: {'成功' if success else '失败'} (状态码: {result.get('status')})")
    
    return {
        "status": result.get("status"),
        "success": success,
        "error": result.get("error", "")
    }


async def test_invalid_model() -> Dict:
    """测试无效模型名"""
    print(f"\n📊 测试无效模型名")
    
    payload = {
        "model": "nonexistent-model-xyz",
        "messages": [{"role": "user", "content": "Hi"}],
        "max_tokens": 10
    }
    
    result = await call_api("/v1/chat/completions", payload, timeout=30)
    
    # 应该返回错误
    is_error_response = result.get("status") in [400, 404, 500]
    print(f"  ✅ 无效模型处理: {'正确拒绝' if is_error_response else '未正确处理'} (状态码: {result.get('status')})")
    
    return {
        "status": result.get("status"),
        "expected_error": True,
        "correctly_rejected": is_error_response
    }


async def test_timeout(model_name: str) -> Dict:
    """测试超时处理"""
    print(f"\n📊 测试超时处理")
    
    # 设置极短的超时
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Write a very long story about space exploration and the universe. Include many details."}],
        "max_tokens": 5000,
        "timeout": 1  # 1秒超时 - 但API目前不支持这个参数
    }
    
    start_time = time.time()
    result = await call_api("/v1/chat/completions", payload, timeout=5)
    elapsed = time.time() - start_time
    
    # API 层面超时测试
    print(f"  ✅ 超时测试: 请求在 {elapsed:.1f}s {'内完成' if elapsed < 5 else '超时'}")
    
    return {
        "elapsed_sec": round(elapsed, 2),
        "completed": result.get("status") == 200
    }


async def test_model_switch(model_name1: str, model_name2: str) -> Dict:
    """测试模型切换"""
    print(f"\n📊 测试模型切换: {model_name1} -> {model_name2}")
    
    # 加载第一个模型
    payload1 = {
        "model": model_name1,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    await call_api("/v1/chat/completions", payload1, timeout=300)
    await asyncio.sleep(2)
    
    # 切换到第二个模型
    payload2 = {
        "model": model_name2,
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10
    }
    
    start_time = time.time()
    result = await call_api("/v1/chat/completions", payload2, timeout=300)
    switch_time = time.time() - start_time
    
    success = result.get("status") == 200
    print(f"  ✅ 模型切换: {'成功' if success else '失败'}, 耗时 {switch_time:.1f}s")
    
    return {
        "from_model": model_name1,
        "to_model": model_name2,
        "switch_time_sec": round(switch_time, 1),
        "success": success
    }


async def test_stress(model_name: str, num_requests: int = 10) -> Dict:
    """压力测试 - 连续请求"""
    print(f"\n📊 压力测试: 连续 {num_requests} 次请求")
    
    mem_before = get_memory_usage()
    success_count = 0
    times = []
    
    for i in range(num_requests):
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": f"Tell me a fact about #{i+1}"}],
            "max_tokens": 50
        }
        
        start = time.time()
        result = await call_api("/v1/chat/completions", payload, timeout=120)
        elapsed = time.time() - start
        
        if result.get("status") == 200:
            success_count += 1
        times.append(elapsed)
        
        print(f"  请求 {i+1}/{num_requests}: {'✅' if result.get('status') == 200 else '❌'} ({elapsed:.1f}s)")
        await asyncio.sleep(0.5)  # 短暂间隔
    
    mem_after = get_memory_usage()
    mem_delta = mem_after - mem_before
    
    print(f"  ✅ 压力测试完成: {success_count}/{num_requests} 成功, 内存增量: {mem_delta:.1f}MB")
    
    return {
        "num_requests": num_requests,
        "success_count": success_count,
        "success_rate": f"{success_count/num_requests*100:.0f}%",
        "memory_delta_mb": round(mem_delta, 1),
        "avg_time_sec": round(sum(times)/len(times), 2) if times else 0,
        "min_time_sec": round(min(times), 2) if times else 0,
        "max_time_sec": round(max(times), 2) if times else 0
    }


async def main():
    print("=" * 60)
    print("🚀 MLX Service 性能与边界测试")
    print("=" * 60)
    
    # 检查服务健康
    print("\n📡 检查服务状态...")
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{BASE_URL}/health") as resp:
            health = await resp.json()
            print(f"  ✅ 服务健康: {health}")
    
    # 获取系统内存
    sys_mem = get_system_memory()
    print(f"  💾 系统内存: {sys_mem['used_gb']:.1f}GB / {sys_mem['total_gb']:.1f}GB ({sys_mem['percent']:.1f}%)")
    
    # ========== 性能测试 ==========
    print("\n" + "=" * 60)
    print("📈 性能测试")
    print("=" * 60)
    
    # 测试模型加载时间
    report["performance"]["qwen3.5-122b_load"] = await test_model_load_time("qwen3.5-122b")
    await asyncio.sleep(5)
    
    report["performance"]["qwen3.5-27b_load"] = await test_model_load_time("qwen3.5-27b")
    await asyncio.sleep(5)
    
    # 测试生成速度
    report["performance"]["qwen3.5-122b_generation"] = await test_generation_speed("qwen3.5-122b", max_tokens=200)
    await asyncio.sleep(3)
    
    report["performance"]["qwen3.5-27b_generation"] = await test_generation_speed("qwen3.5-27b", max_tokens=200)
    await asyncio.sleep(3)
    
    # 并发测试
    report["performance"]["concurrent"] = await test_concurrent_requests("qwen3.5-27b", num_requests=3)
    await asyncio.sleep(3)
    
    # ========== 边界测试 ==========
    print("\n" + "=" * 60)
    print("🔬 边界测试")
    print("=" * 60)
    
    report["boundary"]["long_input"] = await test_long_input("qwen3.5-27b")
    await asyncio.sleep(2)
    
    report["boundary"]["empty_input"] = await test_empty_input("qwen3.5-27b")
    await asyncio.sleep(2)
    
    report["boundary"]["invalid_model"] = await test_invalid_model()
    await asyncio.sleep(2)
    
    report["boundary"]["timeout"] = await test_timeout("qwen3.5-27b")
    await asyncio.sleep(2)
    
    report["boundary"]["model_switch"] = await test_model_switch("qwen3.5-27b", "qwen3.5-122b")
    await asyncio.sleep(3)
    
    # ========== 压力测试 ==========
    print("\n" + "=" * 60)
    print("🔥 压力测试")
    print("=" * 60)
    
    report["stress"]["continuous_requests"] = await test_stress("qwen3.5-27b", num_requests=10)
    
    # 内存泄漏检测
    mem_final = get_memory_usage()
    report["stress"]["memory_leak_check"] = {
        "initial_memory_mb": round(report.get("_initial_memory", 0), 1),
        "final_memory_mb": round(mem_final, 1),
        "leak_detected": mem_final > report.get("_initial_memory", 0) + 500  # 500MB 阈值
    }
    
    # ========== 生成报告 ==========
    print("\n" + "=" * 60)
    print("📋 测试报告摘要")
    print("=" * 60)
    
    # 分析问题
    for category, tests in [("performance", report["performance"]), 
                             ("boundary", report["boundary"]), 
                             ("stress", report["stress"])]:
        for name, result in tests.items():
            if isinstance(result, dict):
                if not result.get("success", True) and category != "boundary":
                    report["issues"].append(f"{category}.{name}: 测试失败")
                if result.get("error"):
                    report["issues"].append(f"{category}.{name}: {result.get('error')}")
    
    # 打印关键指标
    print(f"\n🔹 性能指标:")
    for k, v in report["performance"].items():
        if isinstance(v, dict) and "load_time_sec" in v:
            print(f"   {k}: 加载 {v.get('load_time_sec')}s, 内存 +{v.get('memory_delta_mb')}MB")
        if isinstance(v, dict) and "tokens_per_sec" in v:
            print(f"   {k}: {v.get('tokens_per_sec')} tokens/s")
    
    print(f"\n🔹 边界测试:")
    for k, v in report["boundary"].items():
        status = "✅" if v.get("success") or v.get("correctly_rejected") else "❌"
        print(f"   {status} {k}")
    
    print(f"\n🔹 压力测试:")
    stress_result = report["stress"].get("continuous_requests", {})
    print(f"   成功率: {stress_result.get('success_rate', 'N/A')}")
    print(f"   内存变化: {stress_result.get('memory_delta_mb', 'N/A')}MB")
    
    if report["issues"]:
        print(f"\n⚠️  发现问题:")
        for issue in report["issues"]:
            print(f"   - {issue}")
    
    # 保存报告
    report_file = f"/Users/xiphis/mlx-service/tests/benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n📄 详细报告已保存: {report_file}")
    
    return report


if __name__ == "__main__":
    # 安装依赖
    subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp", "psutil", "-q"])
    
    # 运行测试
    asyncio.run(main())