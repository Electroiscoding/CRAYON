"""
XERV CRAYON V4.2.7 - Production Omni-Backend Tokenizer
=======================================================
Copy this ENTIRE script into a Google Colab cell and run it.

IMPORTANT: Enable GPU runtime first:
Runtime -> Change runtime type -> GPU (T4/V100/A100)
"""

import subprocess
import sys
import os
import time

print("=" * 70)
print("XERV CRAYON V4.2.7 INSTALLATION")
print("=" * 70)

# 1. Environment Check
print("[1/7] Checking environment...")
try:
    import torch
    print(f"      PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"      CUDA: {torch.version.cuda} ({torch.cuda.get_device_name(0)})")
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


# 2. Build Dependencies
print("\n[2/7] Installing build dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ninja", "packaging", "wheel", "setuptools>=68.0"])
print("      Done (ninja, packaging, wheel)")


# 3. Clean Old State
print("\n[3/7] Cleaning previous installations...")
os.system("pip uninstall -y xerv-crayon crayon 2>/dev/null")
os.system("rm -rf /tmp/crayon* build dist src/*.egg-info 2>/dev/null")


# 4. Clone Source
print("\n[4/7] Cloning source code...")
timestamp = int(time.time())
clone_dir = f"/tmp/crayon_{timestamp}"
cmd = f"git clone --depth 1 https://github.com/Electroiscoding/CRAYON.git {clone_dir}"
if os.system(cmd) != 0:
    print("      FATAL: Git clone failed!")
    sys.exit(1)

# Verify source
v_check = subprocess.run(["grep", "-m1", "__version__", f"{clone_dir}/src/crayon/__init__.py"], 
                        capture_output=True, text=True)
print(f"      {v_check.stdout.strip()}")


# 5. Build & Install (Streaming Output)
print("\n[5/7] Compiling and Installing (Streaming Logs)...")
print("-" * 70)

build_env = os.environ.copy()
build_env["MAX_JOBS"] = "1"      # Force serial build to prevent OOM
build_env["cuda_home"] = "/usr/local/cuda"

# Stream output line-by-line
cmd = [sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation", clone_dir]
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=build_env, text=True)

# Print output while running
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


# 6. Verification
print("\n[6/7] Verifying installation...")
# Reset module cache
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


# 7. Benchmarks
print("\n" + "=" * 70)
print("BENCHMARKS & TESTING")
print("=" * 70)

from crayon import CrayonVocab

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")
print(f"\nActive Device: {vocab.device.upper()}")
print(f"Backend: {vocab.backend_name}")

if vocab.device == "cpu" and backends.get("cuda"):
    print("NOTE: Running on CPU but CUDA is available. Use device='cuda' to force.")

# Throughput test
text = "The quick brown fox jumps over the lazy dog."
batch_sizes = [1000, 10000, 50000]
print("\nBatch Throughput:")
for bs in batch_sizes:
    batch = [text] * bs
    # Warmup
    vocab.tokenize(batch[:10]) 
    
    start = time.time()
    res = vocab.tokenize(batch)
    dur = time.time() - start
    
    toks = sum(len(x) for x in res)
    print(f"  {bs:>8,} docs: {bs/dur:>12,.0f} docs/sec | {toks/dur:>14,.0f} tokens/sec")

print("\nDone!")
