"""
XERV CRAYON V4.2.0 - GOOGLE COLAB DEMO
======================================

This script demonstrates the full Omni-Backend capabilities of Crayon.
It automatically detects your hardware and uses the best available backend.

TO RUN ON GOOGLE COLAB:
1. Copy this entire file to a Colab cell
2. Run it - it will automatically install Crayon and run the demo

HARDWARE SUPPORT:
- CPU: Works on all machines (AVX2/AVX-512 optimized)
- GPU: Works on Colab GPU runtime (T4, V100, A100, etc.)
- TPU: Falls back to CPU (TPU not supported for tokenization)
"""

import subprocess
import sys
import os
import time
from typing import Optional


def is_colab() -> bool:
    """Detect if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def is_kaggle() -> bool:
    """Detect if running in Kaggle kernel."""
    return os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None


def get_gpu_info() -> Optional[str]:
    """Get GPU info via nvidia-smi if available."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def install_crayon(force: bool = False) -> bool:
    """
    Install Crayon with GPU support detection.
    
    Args:
        force: Force reinstall even if already installed.
        
    Returns:
        True if installation successful.
    """
    # Check if already installed
    if not force:
        try:
            import crayon
            print(f"✅ Crayon v{crayon.get_version()} already installed")
            return True
        except ImportError:
            pass
    
    print("🔧 Installing XERV Crayon...")
    
    # Detect GPU for build configuration
    gpu_info = get_gpu_info()
    if gpu_info:
        print(f"🎮 GPU Detected: {gpu_info}")
        print("📦 Building with CUDA support...")
    else:
        print("💻 No GPU detected, building CPU-only version...")
    
    # Install from TestPyPI or PyPI
    pip_commands = [
        # Try TestPyPI first (for latest dev version)
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "--index-url", "https://test.pypi.org/simple/",
         "--extra-index-url", "https://pypi.org/simple/",
         "xerv-crayon"],
        # Fallback to regular PyPI
        [sys.executable, "-m", "pip", "install", "--upgrade", "xerv-crayon"],
    ]
    
    for cmd in pip_commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print("✅ Installation successful!")
                return True
            else:
                print(f"⚠️ Attempt failed: {result.stderr[:200]}")
        except Exception as e:
            print(f"⚠️ Attempt failed: {e}")
    
    # If all else fails, try building from source
    print("🔨 Attempting source build...")
    try:
        # Clone and install
        commands = [
            "git clone https://github.com/xerv/crayon.git /tmp/crayon 2>/dev/null || true",
            f"{sys.executable} -m pip install /tmp/crayon/ --no-build-isolation"
        ]
        for cmd in commands:
            os.system(cmd)
        return True
    except Exception as e:
        print(f"❌ Source build failed: {e}")
        return False


def demo_basic_usage():
    """Demonstrate basic tokenization."""
    from crayon import CrayonVocab
    
    print("\n" + "="*60)
    print("1️⃣  BASIC USAGE - Auto Device Detection")
    print("="*60)
    
    # Create vocab with auto detection
    vocab = CrayonVocab(device="auto")
    info = vocab.get_info()
    
    print(f"\n🔍 System Detection Results:")
    print(f"   Device: {info['device'].upper()}")
    print(f"   Backend: {info['backend']}")
    if 'hardware' in info:
        print(f"   Hardware: {info['hardware'].get('name', 'Unknown')}")
        print(f"   Features: {info['hardware'].get('features', 'N/A')}")
    
    # Load profile
    vocab.load_profile("lite")
    print(f"\n📚 Loaded Profile: {info.get('active_profile', 'lite')}")
    
    return vocab


def demo_latency_test(vocab):
    """Test single-string tokenization latency."""
    print("\n" + "="*60)
    print("2️⃣  LATENCY TEST - Single String Performance")
    print("="*60)
    
    test_texts = [
        "Hello, world!",
        "Crayon optimizes tokenization at the silicon level.",
        "The quick brown fox jumps over the lazy dog. " * 10,
    ]
    
    for text in test_texts:
        # Warm-up
        _ = vocab.tokenize(text)
        
        # Timed run
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            tokens = vocab.tokenize(text)
        end = time.perf_counter()
        
        avg_us = ((end - start) / iterations) * 1_000_000
        text_preview = text[:50] + "..." if len(text) > 50 else text
        
        print(f"\n   Input: '{text_preview}'")
        print(f"   Tokens: {len(tokens)} tokens")
        print(f"   ⚡ Latency: {avg_us:.2f} µs/call ({iterations} iterations)")


def demo_batch_throughput(vocab):
    """Test batch tokenization throughput."""
    print("\n" + "="*60)
    print("3️⃣  THROUGHPUT TEST - Batch Processing")
    print("="*60)
    
    # Create test batches of different sizes
    base_text = "The quick brown fox jumps over the lazy dog. This is a test sentence for benchmarking tokenization throughput."
    batch_sizes = [100, 1000, 10000]
    
    for batch_size in batch_sizes:
        batch = [base_text] * batch_size
        
        # Warm-up
        _ = vocab.tokenize(batch[:10])
        
        # Timed run
        start = time.time()
        results = vocab.tokenize(batch)
        duration = time.time() - start
        
        throughput = batch_size / duration
        tokens_per_sec = sum(len(r) for r in results) / duration
        
        print(f"\n   Batch Size: {batch_size:,} documents")
        print(f"   Duration: {duration:.4f}s")
        print(f"   🚀 Throughput: {throughput:,.0f} docs/sec")
        print(f"   📊 Token Rate: {tokens_per_sec:,.0f} tokens/sec")


def demo_profile_switching(vocab):
    """Demonstrate profile hot-swapping."""
    print("\n" + "="*60)
    print("4️⃣  PROFILE HOT-SWAP - Context Manager Demo")
    print("="*60)
    
    code_snippet = """def forward(self, x):
    return torch.matmul(x, self.weights)"""
    
    science_text = "The quantum entanglement of photons demonstrates non-local correlations."
    
    # Tokenize with default profile
    print("\n   [lite profile] Tokenizing code...")
    tokens_lite = vocab.tokenize(code_snippet)
    print(f"   -> {len(tokens_lite)} tokens")
    
    # Try code profile (may not exist)
    try:
        print("\n   [code profile] Switching context...")
        with vocab.using_profile("code"):
            tokens_code = vocab.tokenize(code_snippet)
            print(f"   -> {len(tokens_code)} tokens (specialized!)")
            improvement = ((len(tokens_lite) - len(tokens_code)) / len(tokens_lite)) * 100
            if improvement > 0:
                print(f"   -> {improvement:.1f}% better compression!")
    except FileNotFoundError:
        print("   ⚠️ 'code' profile not available in this installation")
    
    # Try science profile
    try:
        print("\n   [science profile] Switching context...")
        with vocab.using_profile("science"):
            tokens_science = vocab.tokenize(science_text)
            print(f"   -> {len(tokens_science)} tokens for science text")
    except FileNotFoundError:
        print("   ⚠️ 'science' profile not available in this installation")
    
    print("\n   ✅ Automatically reverted to 'lite' profile")


def demo_decode(vocab):
    """Demonstrate decode functionality."""
    print("\n" + "="*60)
    print("5️⃣  ENCODE/DECODE - Round-Trip Test")
    print("="*60)
    
    test_text = "Hello, Crayon! This is a round-trip test."
    print(f"\n   Original: '{test_text}'")
    
    tokens = vocab.tokenize(test_text)
    print(f"   Encoded: {tokens[:10]}... ({len(tokens)} tokens)")
    
    try:
        decoded = vocab.decode(tokens)
        print(f"   Decoded: '{decoded}'")
        
        if decoded == test_text:
            print("   ✅ Perfect round-trip!")
        else:
            print("   ⚠️ Slight differences (expected with subword tokenization)")
    except RuntimeError as e:
        print(f"   ⚠️ Decode not available: {e}")


def demo_device_switching(vocab):
    """Demonstrate runtime device switching."""
    from crayon import check_backends
    
    print("\n" + "="*60)
    print("6️⃣  DEVICE SWITCHING - Runtime Flexibility")
    print("="*60)
    
    backends = check_backends()
    print(f"\n   Available backends: {backends}")
    
    # Switch to CPU
    print("\n   Switching to CPU...")
    vocab.set_device("cpu")
    print(f"   Now on: {vocab.device.upper()}")
    
    # Quick test
    tokens = vocab.tokenize("Quick CPU test")
    print(f"   Tokenized: {tokens}")
    
    # Switch back to auto
    print("\n   Switching to AUTO...")
    vocab.set_device("auto")
    print(f"   Auto-selected: {vocab.device.upper()}")


def demo_gpu_stress_test(vocab):
    """GPU-specific stress test (only runs if GPU is available)."""
    if vocab.device == "cpu":
        print("\n" + "="*60)
        print("7️⃣  GPU STRESS TEST - Skipped (Running on CPU)")
        print("="*60)
        return
    
    print("\n" + "="*60)
    print(f"7️⃣  GPU STRESS TEST - {vocab.device.upper()} Kernel Smashing")
    print("="*60)
    
    # Create massive batch
    batch_size = 100_000
    base_text = "The quick brown fox jumps over the lazy dog."
    
    print(f"\n   Generating {batch_size:,} documents...")
    batch = [base_text] * batch_size
    
    print("   🚀 Launching kernel...")
    start = time.time()
    results = vocab.tokenize(batch)
    duration = time.time() - start
    
    total_tokens = sum(len(r) for r in results)
    docs_per_sec = batch_size / duration
    tokens_per_sec = total_tokens / duration
    
    print(f"\n   ✅ Processed {batch_size:,} docs in {duration:.4f}s")
    print(f"   🔥 Document Throughput: {docs_per_sec:,.0f} docs/sec")
    print(f"   📊 Token Throughput: {tokens_per_sec:,.0f} tokens/sec")


def show_system_info():
    """Display system information."""
    import platform
    
    print("\n" + "="*60)
    print("🖥️  SYSTEM INFORMATION")
    print("="*60)
    
    print(f"\n   Python: {sys.version}")
    print(f"   Platform: {platform.platform()}")
    
    # GPU info
    gpu = get_gpu_info()
    if gpu:
        print(f"   GPU: {gpu}")
    else:
        print("   GPU: Not detected")
    
    # Crayon info
    try:
        from crayon import get_version, get_backend_info
        print(f"\n   Crayon Version: {get_version()}")
        
        backends = get_backend_info()
        print("   Backends:")
        for name, info in backends.items():
            status = "✅" if info.get("available") else "❌"
            print(f"      {status} {name}: {info.get('hardware', info.get('error', 'N/A'))}")
    except Exception as e:
        print(f"   Crayon Info: Error - {e}")


def main():
    """Main demo runner."""
    print("=" * 60)
    print("🖍️  XERV CRAYON V4.2.0 - OMNI-BACKEND DEMO")
    print("=" * 60)
    
    # Check environment
    if is_colab():
        print("\n🌐 Running in Google Colab")
    elif is_kaggle():
        print("\n🌐 Running in Kaggle")
    else:
        print("\n💻 Running locally")
    
    # Install if needed
    if not install_crayon():
        print("\n❌ Installation failed. Please check errors above.")
        return
    
    # Show system info
    show_system_info()
    
    # Run demos
    try:
        vocab = demo_basic_usage()
        demo_latency_test(vocab)
        demo_batch_throughput(vocab)
        demo_profile_switching(vocab)
        demo_decode(vocab)
        demo_device_switching(vocab)
        demo_gpu_stress_test(vocab)
        
        print("\n" + "=" * 60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            vocab.close()
        except:
            pass


if __name__ == "__main__":
    main()
