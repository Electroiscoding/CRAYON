"""
XERV CRAYON Local Benchmark Suite
==================================
Comprehensive hardware detection and performance benchmarking
"""

import time
import platform
import subprocess
import sys
from typing import Dict, List, Tuple

def detect_hardware() -> Dict:
    """Deep hardware detection for CPU and GPU"""
    hw_info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "python": platform.python_version(),
        "cpu": {},
        "gpu": {}
    }
    
    if platform.system() == "Windows":
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "name"],
                capture_output=True,
                text=True,
                timeout=5
            )
            cpu_name = result.stdout.strip().split('\n')[1].strip()
            hw_info["cpu"]["name"] = cpu_name
        except:
            hw_info["cpu"]["name"] = platform.processor()
        
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "NumberOfCores"],
                capture_output=True,
                text=True,
                timeout=5
            )
            cores = result.stdout.strip().split('\n')[1].strip()
            hw_info["cpu"]["cores"] = int(cores)
        except:
            hw_info["cpu"]["cores"] = "Unknown"
        
        try:
            result = subprocess.run(
                ["wmic", "cpu", "get", "MaxClockSpeed"],
                capture_output=True,
                text=True,
                timeout=5
            )
            freq = result.stdout.strip().split('\n')[1].strip()
            hw_info["cpu"]["frequency_mhz"] = int(freq)
        except:
            hw_info["cpu"]["frequency_mhz"] = "Unknown"
    else:
        try:
            result = subprocess.run(
                ["lscpu"],
                capture_output=True,
                text=True,
                timeout=5
            )
            for line in result.stdout.split('\n'):
                if "Model name:" in line:
                    hw_info["cpu"]["name"] = line.split(':')[1].strip()
                elif "CPU(s):" in line and "NUMA" not in line:
                    hw_info["cpu"]["cores"] = line.split(':')[1].strip()
                elif "CPU MHz:" in line:
                    hw_info["cpu"]["frequency_mhz"] = float(line.split(':')[1].strip())
        except:
            hw_info["cpu"]["name"] = platform.processor()
    
    try:
        import torch
        hw_info["pytorch"] = torch.__version__
        
        if torch.cuda.is_available():
            hw_info["gpu"]["available"] = True
            hw_info["gpu"]["count"] = torch.cuda.device_count()
            hw_info["gpu"]["devices"] = []
            
            for i in range(torch.cuda.device_count()):
                device_info = {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "capability": torch.cuda.get_device_capability(i),
                    "total_memory_gb": torch.cuda.get_device_properties(i).total_memory / 1e9
                }
                hw_info["gpu"]["devices"].append(device_info)
            
            hw_info["gpu"]["cuda_version"] = torch.version.cuda
        else:
            hw_info["gpu"]["available"] = False
    except ImportError:
        hw_info["pytorch"] = "Not installed"
        hw_info["gpu"]["available"] = False
    
    try:
        result = subprocess.run(
            ["nvcc", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if "release" in line.lower():
                    hw_info["nvcc_version"] = line.strip()
                    break
    except:
        hw_info["nvcc_version"] = "Not found"
    
    return hw_info

def print_hardware_info(hw_info: Dict):
    """Print formatted hardware information"""
    print("=" * 70)
    print("HARDWARE DETECTION")
    print("=" * 70)
    
    print(f"\n[*] System Information:")
    print(f"   OS: {hw_info['os']} {hw_info['os_version']}")
    print(f"   Python: {hw_info['python']}")
    if "pytorch" in hw_info:
        print(f"   PyTorch: {hw_info['pytorch']}")
    
    print(f"\n[*] CPU Information:")
    cpu = hw_info.get("cpu", {})
    print(f"   Model: {cpu.get('name', 'Unknown')}")
    print(f"   Cores: {cpu.get('cores', 'Unknown')}")
    if "frequency_mhz" in cpu:
        freq = cpu["frequency_mhz"]
        if isinstance(freq, (int, float)):
            print(f"   Frequency: {freq:.0f} MHz ({freq/1000:.2f} GHz)")
        else:
            print(f"   Frequency: {freq}")
    
    if hw_info.get("gpu", {}).get("available"):
        print(f"\n[*] GPU Information:")
        for device in hw_info["gpu"]["devices"]:
            print(f"   Device {device['id']}: {device['name']}")
            print(f"      Compute Capability: {device['capability'][0]}.{device['capability'][1]}")
            print(f"      Memory: {device['total_memory_gb']:.2f} GB")
        print(f"   CUDA Version: {hw_info['gpu']['cuda_version']}")
        if "nvcc_version" in hw_info:
            print(f"   NVCC: {hw_info['nvcc_version']}")
    else:
        print(f"\n[*] GPU: Not available")
    
    print()

def run_crayon_benchmarks() -> Dict:
    """Run comprehensive CRAYON benchmarks"""
    print("=" * 70)
    print("XERV CRAYON BENCHMARKS")
    print("=" * 70)
    
    try:
        from crayon import CrayonVocab, check_backends
    except ImportError:
        print("\n❌ ERROR: CRAYON not installed!")
        print("   Run: pip install -e .")
        sys.exit(1)
    
    backends = check_backends()
    print(f"\nAvailable Backends: {backends}")
    
    results = {}
    test_text = "The quick brown fox jumps over the lazy dog."
    batch_sizes = [1000, 10000, 50000]
    
    for device in ["cpu", "cuda"]:
        if not backends.get(device):
            continue
        
        print(f"\n{'-' * 70}")
        print(f"Testing {device.upper()} Backend")
        print(f"{'-' * 70}")
        
        try:
            vocab = CrayonVocab(device=device)
            vocab.load_profile("lite")
            
            info = vocab.get_info()
            print(f"Backend: {info['backend']}")
            if 'profile' in info:
                print(f"Profile: {info['profile']}")
            print(f"Vocab Size: {info['vocab_size']:,}")
            
            device_results = []
            print(f"\nBatch Throughput ({device.upper()}):")
            
            for bs in batch_sizes:
                batch = [test_text] * bs
                
                vocab.tokenize(batch[:10])
                
                start = time.time()
                res = vocab.tokenize(batch)
                dur = time.time() - start
                
                total_tokens = sum(len(x) for x in res)
                docs_per_sec = bs / dur
                tokens_per_sec = total_tokens / dur
                
                device_results.append({
                    "batch_size": bs,
                    "docs_per_sec": docs_per_sec,
                    "tokens_per_sec": tokens_per_sec,
                    "duration": dur
                })
                
                print(f"   {bs:>8,} docs: {docs_per_sec:>12,.0f} docs/sec | {tokens_per_sec:>14,.0f} tokens/sec")
            
            results[device] = device_results
            
        except Exception as e:
            print(f"   [ERROR] Error testing {device}: {e}")
    
    return results

def run_tiktoken_benchmark() -> Dict:
    """Run tiktoken benchmark for comparison"""
    print(f"\n{'=' * 70}")
    print("TIKTOKEN BENCHMARK (Comparison)")
    print("=" * 70)
    
    try:
        import tiktoken
    except ImportError:
        print("\n[!] Tiktoken not installed, skipping comparison")
        print("   Install with: pip install tiktoken")
        return {}
    
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        test_text = "The quick brown fox jumps over the lazy dog."
        batch_sizes = [1000, 10000, 50000]
        
        results = []
        print(f"\nTiktoken Batch Throughput (cl100k_base):")
        
        for bs in batch_sizes:
            batch = [test_text] * bs
            
            enc.encode_batch([test_text] * 10)
            
            start = time.time()
            res = enc.encode_batch(batch)
            dur = time.time() - start
            
            total_tokens = sum(len(x) for x in res)
            docs_per_sec = bs / dur
            tokens_per_sec = total_tokens / dur
            
            results.append({
                "batch_size": bs,
                "docs_per_sec": docs_per_sec,
                "tokens_per_sec": tokens_per_sec
            })
            
            print(f"   {bs:>8,} docs: {docs_per_sec:>12,.0f} docs/sec | {tokens_per_sec:>14,.0f} tokens/sec")
        
        return {"tiktoken": results}
        
    except Exception as e:
        print(f"   [ERROR] {e}")
        return {}

def print_summary(crayon_results: Dict, tiktoken_results: Dict):
    """Print benchmark summary comparison"""
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    if not crayon_results:
        print("\n[!] No CRAYON results to display")
        return
    
    print("\nPerformance Comparison:")
    print("-" * 95)
    print(f"{'Batch Size':<15} | {'CRAYON Docs/Sec':<20} | {'CRAYON Tokens/Sec':<20} | {'Tiktoken Docs/Sec':<20} | {'Tiktoken Tokens/Sec':<20}")
    print("-" * 95)
    
    device = "cuda" if "cuda" in crayon_results else "cpu"
    crayon_data = crayon_results[device]
    tiktoken_data = tiktoken_results.get("tiktoken", [])
    
    for i, result in enumerate(crayon_data):
        bs = result["batch_size"]
        crayon_docs = f"{result['docs_per_sec']:,.0f}"
        crayon_tokens = f"{result['tokens_per_sec']:,.0f}"
        
        if i < len(tiktoken_data):
            tik_docs = f"{tiktoken_data[i]['docs_per_sec']:,.0f}"
            tik_tokens = f"{tiktoken_data[i]['tokens_per_sec']:,.0f}"
        else:
            tik_docs = "N/A"
            tik_tokens = "N/A"
        
        print(f"{bs:<15,} | {crayon_docs:<20} | {crayon_tokens:<20} | {tik_docs:<20} | {tik_tokens:<20}")
    
    print("-" * 95)
    
    if tiktoken_data:
        avg_crayon = sum(r["tokens_per_sec"] for r in crayon_data) / len(crayon_data)
        avg_tiktoken = sum(r["tokens_per_sec"] for r in tiktoken_data) / len(tiktoken_data)
        speedup = avg_crayon / avg_tiktoken
        
        print(f"\n[*] Average Speedup: {speedup:.1f}x faster than tiktoken")
        print(f"   CRAYON ({device.upper()}): {avg_crayon:,.0f} tokens/sec")
        print(f"   Tiktoken: {avg_tiktoken:,.0f} tokens/sec")

def main():
    """Main benchmark execution"""
    print("\n" + "=" * 70)
    print("XERV CRAYON V4.1.9 - LOCAL BENCHMARK SUITE")
    print("=" * 70)
    
    hw_info = detect_hardware()
    print_hardware_info(hw_info)
    
    crayon_results = run_crayon_benchmarks()
    
    tiktoken_results = run_tiktoken_benchmark()
    
    print_summary(crayon_results, tiktoken_results)
    
    print("\n" + "=" * 70)
    print("[*] Benchmark Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
