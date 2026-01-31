"""
XERV CRAYON V4.1.9 - Google Colab Installation and Benchmark Script
====================================================================
This script installs CRAYON from GitHub and runs comprehensive benchmarks
on Google Colab's GPU infrastructure (T4/V100/A100).

Usage:
    1. Open Google Colab
    2. Runtime -> Change runtime type -> GPU (T4 recommended)
    3. Copy this entire file into a cell and run
"""

import subprocess
import sys
import os
import time

def print_section(title: str, char: str = "="):
    """Print formatted section header"""
    print(f"\n{char * 70}")
    print(title)
    print(f"{char * 70}\n")

def run_command(cmd, description: str = None, stream: bool = False):
    """Execute shell command with optional output streaming"""
    if description:
        print(f"▶ {description}")
    
    if stream:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            shell=isinstance(cmd, str)
        )
        
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(line.rstrip())
        
        return process.poll()
    else:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            shell=isinstance(cmd, str)
        )
        return result.returncode

print_section("XERV CRAYON V4.1.9 INSTALLATION AND BENCHMARKS")

print("[1/7] Checking environment...")
try:
    import torch
    print(f"      PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        print(f"      CUDA: {cuda_version} ({device_name})")
        print("      * Smart Build: Will compile ONLY for this GPU architecture")
    else:
        print("      CUDA: Not available (CPU only)")
except ImportError:
    print("      PyTorch not found (will be installed)")

nvcc_check = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
if nvcc_check.returncode == 0:
    print(f"      NVCC: {nvcc_check.stdout.strip()}")
else:
    print("      NVCC: Not found")

print("\n[2/7] Installing build dependencies...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "ninja", "packaging", "wheel", "setuptools>=68.0"
])
print("      Done (ninja, packaging, wheel)")

print("\n[3/7] Cleaning previous installations...")
os.system("pip uninstall -y xerv-crayon crayon 2>/dev/null")
os.system("rm -rf /tmp/crayon* build dist src/*.egg-info 2>/dev/null")

print("\n[4/7] Cloning source code...")
timestamp = int(time.time())
clone_dir = f"/tmp/crayon_{timestamp}"
cmd = f"git clone --depth 1 https://github.com/Electroiscoding/CRAYON.git {clone_dir}"
if os.system(cmd) != 0:
    print("      FATAL: Git clone failed!")
    sys.exit(1)

v_check = subprocess.run(
    ["grep", "-m1", "__version__", f"{clone_dir}/src/crayon/__init__.py"],
    capture_output=True,
    text=True
)
print(f"      {v_check.stdout.strip()}")

print("\n[5/7] Compiling and Installing (Streaming Logs)...")
print("-" * 70)

build_env = os.environ.copy()
build_env["MAX_JOBS"] = "1"
build_env["CUDA_HOME"] = "/usr/local/cuda"

cmd = [sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation", clone_dir]
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env=build_env,
    text=True
)

while True:
    line = process.stdout.readline()
    if not line and process.poll() is not None:
        break
    if line:
        print(line.rstrip())

rc = process.poll()
print("-" * 70)

if rc != 0:
    print("\n" + "!" * 70)
    print("FATAL ERROR: Installation failed!")
    print(f"Exit Code: {rc}")
    print("!" * 70)
    sys.exit(1)

print("\n[6/7] Verifying installation...")
for key in list(sys.modules.keys()):
    if "crayon" in key:
        del sys.modules[key]

try:
    import crayon
    print(f"      Success! Installed version: {crayon.get_version()}")
    backends = crayon.check_backends()
    print(f"      Backends: {backends}")
except ImportError as e:
    print(f"      FATAL: Could not import crayon: {e}")
    sys.exit(1)

print_section("XERV CRAYON BENCHMARKS")

from crayon import CrayonVocab

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")
print(f"Active Device: {vocab.device.upper()}")

info = vocab.get_info()
print(f"Backend: {info['backend']}")

if vocab.device == "cpu" and backends.get("cuda"):
    print("NOTE: Running on CPU but CUDA is available. Use device='cuda' to force.")

text = "The quick brown fox jumps over the lazy dog."
batch_sizes = [1000, 10000, 50000]

print(f"\nBatch Throughput (XERV CRAYON):")
for bs in batch_sizes:
    batch = [text] * bs
    vocab.tokenize(batch[:10])
    
    start = time.time()
    res = vocab.tokenize(batch)
    dur = time.time() - start
    
    toks = sum(len(x) for x in res)
    print(f"     {bs:>6,} docs: {bs/dur:>12,.0f} docs/sec | {toks/dur:>14,.0f} tokens/sec")

print_section("TIKTOKEN INSTALLATION AND BENCHMARKS")

try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tiktoken"])
    print("Tiktoken installed successfully.\n")
    
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    
    print("Tiktoken Batch Throughput (cl100k_base encoding):")
    for bs in batch_sizes:
        batch = [text] * bs
        enc.encode_batch([text] * 10)
        
        start = time.time()
        res = enc.encode_batch(batch)
        dur = time.time() - start
        
        toks = sum(len(x) for x in res)
        print(f"     {bs:>6,} docs: {bs/dur:>12,.0f} docs/sec | {toks/dur:>14,.0f} tokens/sec")
        
except Exception as e:
    print(f"⚠️  Tiktoken benchmark failed: {e}")

print_section("SUMMARY OF BENCHMARK RESULTS")

print("Done with all installations and benchmarks!")
