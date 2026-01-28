#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XERV CRAYON V4.2.0 - OMNI-BACKEND DEMONSTRATION
================================================

This script demonstrates the "Smashing Experience" of Crayon's Omni-Backend.
It showcases:
1. Automatic hardware detection (Auto-Pilot Mode)
2. Manual device override
3. Profile hot-swapping
4. Latency and throughput benchmarks

Usage:
    python demo_omni.py

The script will automatically detect your hardware and run appropriate tests.
"""

import time
import sys
import os
import io

# Fix Windows console encoding for emoji support
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # If it fails, just continue without emoji

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from crayon import CrayonVocab, check_backends, get_version, enable_verbose_logging


def print_banner():
    """Print the demo banner."""
    print("=" * 70)
    print("🖍️  XERV CRAYON V{} - OMNI-BACKEND DEMO".format(get_version()))
    print("=" * 70)
    print()


def demo_auto_mode():
    """
    AUTO MODE: The "It Just Works" Experience
    
    Crayon automatically detects your hardware and selects the best backend:
    - NVIDIA GPU → CUDA engine (parallel kernel execution)
    - AMD GPU → ROCm engine (HIP kernel execution)
    - Otherwise → CPU engine (AVX2/AVX-512 SIMD)
    """
    print("1️⃣  INITIALIZING IN AUTO MODE...")
    print("-" * 50)
    
    # Enable logging to see device detection
    enable_verbose_logging()
    
    # Create vocab with auto-detection
    vocab = CrayonVocab(device="auto")
    
    info = vocab.get_info()
    print(f"\n   📊 Detection Results:")
    print(f"   ├─ Device: {info['device'].upper()}")
    print(f"   ├─ Backend: {info['backend']}")
    print(f"   ├─ State: {info['device_state']}")
    
    if 'hardware' in info:
        print(f"   └─ Hardware: {info['hardware'].get('name', 'Unknown')}")
        if info['hardware'].get('vram_mb'):
            print(f"      └─ VRAM: {info['hardware']['vram_mb']} MB")
    
    # Show available backends
    backends = check_backends()
    available = [k for k, v in backends.items() if v]
    print(f"\n   🔌 Available Backends: {', '.join(available)}")
    
    # Load default profile
    print("\n   📦 Loading 'lite' profile...")
    vocab.load_profile("lite")
    print(f"   ✅ Profile loaded ({vocab.vocab_size} tokens)")
    
    return vocab


def demo_latency_test(vocab):
    """
    LATENCY TEST: The "Instant" Feel
    
    Measures single-string tokenization performance.
    CPU mode is optimized for latency with minimal overhead.
    """
    print("\n")
    print("2️⃣  LATENCY TEST (Single String)")
    print("-" * 50)
    
    text = "Crayon optimizes tokenization at the silicon level."
    
    # Warm-up (important for JIT and cache warming)
    for _ in range(100):
        _ = vocab.tokenize(text)
    
    # Timed run
    iterations = 10000
    start = time.perf_counter()
    for _ in range(iterations):
        tokens = vocab.tokenize(text)
    end = time.perf_counter()
    
    avg_us = ((end - start) / iterations) * 1_000_000
    
    print(f"\n   📝 Input: '{text}'")
    print(f"   🔢 Tokens: {tokens}")
    print(f"   📊 Token Count: {len(tokens)}")
    print(f"   ⚡ Average Latency: {avg_us:.2f} µs/call")
    print(f"   🔄 Iterations: {iterations:,}")
    
    return tokens


def demo_profile_hotswap(vocab):
    """
    PROFILE HOT-SWAP: The Context Manager
    
    Demonstrates switching vocabulary profiles on-the-fly.
    Useful when processing mixed content (code, science, general text).
    """
    print("\n")
    print("3️⃣  CONTEXT SWITCHING (Profile Hot-Swap)")
    print("-" * 50)
    
    code_snippet = "def forward(self, x): return torch.matmul(x, w)"
    
    print(f"\n   📝 Code: '{code_snippet}'")
    
    # Tokenize with lite profile
    print("\n   [LITE Profile] Tokenizing code...")
    tokens_lite = vocab.tokenize(code_snippet)
    print(f"   └─ Result: {len(tokens_lite)} tokens")
    
    # Try code profile
    try:
        print("\n   [CODE Profile] Switching context...")
        with vocab.using_profile("code"):
            tokens_code = vocab.tokenize(code_snippet)
            print(f"   └─ Result: {len(tokens_code)} tokens")
            
            if len(tokens_code) < len(tokens_lite):
                improvement = ((len(tokens_lite) - len(tokens_code)) / len(tokens_lite)) * 100
                print(f"   ✨ {improvement:.1f}% better compression with specialized profile!")
    except FileNotFoundError:
        print("   ⚠️ 'code' profile not available - using lite only")
    
    print("\n   🔄 Automatically reverted to 'lite' profile")
    
    # Verify we're back to lite
    current_info = vocab.get_info()
    print(f"   └─ Current: {current_info.get('active_profile', 'unknown')}")


def demo_batch_throughput(vocab):
    """
    BATCH THROUGHPUT: The Parallel Processing Power
    
    Measures batch tokenization performance.
    GPU mode excels here with parallel kernel execution.
    """
    print("\n")
    print("4️⃣  BATCH THROUGHPUT TEST")
    print("-" * 50)
    
    # Create test batches
    base_text = "The quick brown fox jumps over the lazy dog."
    batch_sizes = [100, 1000, 10000]
    
    for batch_size in batch_sizes:
        batch = [base_text] * batch_size
        
        # Warm-up
        _ = vocab.tokenize(batch[:10])
        
        # Timed run
        start = time.time()
        results = vocab.tokenize(batch)
        duration = time.time() - start
        
        total_tokens = sum(len(r) for r in results)
        throughput = batch_size / duration
        tokens_per_sec = total_tokens / duration
        
        print(f"\n   📦 Batch Size: {batch_size:,}")
        print(f"   ⏱️  Duration: {duration:.4f}s")
        print(f"   🚀 Throughput: {throughput:,.0f} docs/sec")
        print(f"   📊 Token Rate: {tokens_per_sec:,.0f} tokens/sec")


def demo_gpu_smashing(vocab):
    """
    GPU SMASHING: The High-Throughput Experience
    
    If running on GPU, demonstrates the massive parallelism available.
    100K+ documents processed in seconds.
    """
    print("\n")
    print("5️⃣  GPU SMASH TEST")
    print("-" * 50)
    
    if vocab.device == "cpu":
        print("\n   ℹ️ Running in CPU Mode - Skipping GPU stress test")
        print("   💡 To enable: Run on a machine with NVIDIA/AMD GPU")
        return
    
    # Massive batch
    batch_size = 100_000
    base_text = "The quick brown fox jumps over the lazy dog."
    
    print(f"\n   🔧 Generating {batch_size:,} documents...")
    batch = [base_text] * batch_size
    
    print("   🚀 Launching GPU kernel...")
    start = time.time()
    results = vocab.tokenize(batch)
    duration = time.time() - start
    
    total_tokens = sum(len(r) for r in results)
    throughput = batch_size / duration
    tokens_per_sec = total_tokens / duration
    
    print(f"\n   ✅ Processed {batch_size:,} documents in {duration:.4f}s")
    print(f"   🔥 Document Throughput: {throughput:,.0f} docs/sec")
    print(f"   📊 Token Throughput: {tokens_per_sec:,.0f} tokens/sec")


def demo_encode_decode(vocab):
    """
    ENCODE/DECODE: Round-Trip Verification
    
    Demonstrates the decode() functionality for debugging
    and understanding tokenization behavior.
    """
    print("\n")
    print("6️⃣  ENCODE/DECODE ROUND-TRIP")
    print("-" * 50)
    
    test_text = "Hello, Crayon! Testing the tokenizer."
    print(f"\n   📝 Original: '{test_text}'")
    
    # Encode
    tokens = vocab.tokenize(test_text)
    print(f"   🔢 Tokens: {tokens}")
    
    # Decode (if JSON available)
    try:
        decoded = vocab.decode(tokens)
        print(f"   📤 Decoded: '{decoded}'")
        
        if decoded == test_text:
            print("   ✅ Perfect round-trip!")
        else:
            print("   ⚠️ Minor differences (expected with subword tokenization)")
    except RuntimeError as e:
        print(f"   ⚠️ Decode unavailable: {e}")


def demo_device_override():
    """
    MANUAL OVERRIDE: Total Control
    
    Demonstrates explicitly selecting a device for specific use cases.
    """
    print("\n")
    print("7️⃣  MANUAL DEVICE OVERRIDE")
    print("-" * 50)
    
    backends = check_backends()
    print(f"\n   🔌 Available: {backends}")
    
    # Force CPU mode
    print("\n   🔵 Creating CPU-only instance...")
    cpu_vocab = CrayonVocab(device="cpu")
    cpu_vocab.load_profile("lite")
    
    info = cpu_vocab.get_info()
    print(f"   └─ Device: {info['device']}")
    print(f"   └─ Backend: {info['backend']}")
    
    # Quick latency test
    text = "Quick CPU test"
    start = time.perf_counter()
    for _ in range(1000):
        _ = cpu_vocab.tokenize(text)
    avg_us = ((time.perf_counter() - start) / 1000) * 1_000_000
    print(f"   └─ Latency: {avg_us:.2f} µs/call")
    
    cpu_vocab.close()
    
    # Try CUDA if available
    if backends.get("cuda"):
        print("\n   🟢 Creating CUDA instance...")
        cuda_vocab = CrayonVocab(device="cuda")
        cuda_vocab.load_profile("lite")
        info = cuda_vocab.get_info()
        print(f"   └─ Device: {info['device']}")
        cuda_vocab.close()
    
    # Try ROCm if available
    if backends.get("rocm"):
        print("\n   🔴 Creating ROCm instance...")
        rocm_vocab = CrayonVocab(device="rocm")
        rocm_vocab.load_profile("lite")
        info = rocm_vocab.get_info()
        print(f"   └─ Device: {info['device']}")
        rocm_vocab.close()


def main():
    """Run the complete demo."""
    print_banner()
    
    try:
        # Main demos
        vocab = demo_auto_mode()
        demo_latency_test(vocab)
        demo_profile_hotswap(vocab)
        demo_batch_throughput(vocab)
        demo_gpu_smashing(vocab)
        demo_encode_decode(vocab)
        
        # Cleanup main vocab
        vocab.close()
        
        # Device override demo
        demo_device_override()
        
        print("\n")
        print("=" * 70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
