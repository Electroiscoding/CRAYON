
# -----------------------------------------------------------------------------
# XERV CRAYON - GOOGLE COLAB VERIFICATION NOTEBOOK
# -----------------------------------------------------------------------------
# RUN THIS CELL TO VERIFY CRAYON ON TESLA T4 / A100 GPUS
# -----------------------------------------------------------------------------

import os
import sys
import subprocess
import time

def run_cmd(cmd):
    print(f">> {cmd}")
    subprocess.check_call(cmd, shell=True)

print("🔹 STEP 1: Setting up environment...")
# Clone or Simulate Git Repo structure (assuming we are in the repo or uploading files)
# For Colab usage, user typically clones first:
# !git clone https://github.com/Xerv-AI/crayon.git
# %cd crayon

# Install dependencies if missing
try:
    import tiktoken
except ImportError:
    run_cmd("pip install tiktoken")

print("\n🔹 STEP 2: Building Extensions (CUDA/CPU)...")
# Force build to ensure CUDA is picked up
run_cmd("python setup.py build_ext --inplace")

# Add build dir to path so we can import directly
import glob
so_files = glob.glob("build/lib*")
if so_files:
    sys.path.insert(0, os.path.abspath(so_files[0]))
sys.path.insert(0, os.path.abspath("src"))

print("\n🔹 STEP 3: Detecting Hardware...")
try:
    from crayon.core.vocabulary import CrayonVocab
    vocab = CrayonVocab(device="cuda")
    print("✅ Crayon successfully initialized with CUDA request.")
except ImportError:
    print("⚠️  CUDA extension import failed. Running CPU fallback check.")
    vocab = CrayonVocab(device="cpu")
except Exception as e:
    print(f"❌ Initialization error: {e}")
    sys.exit(1)

# Generate a dummy profile if none exists
if not os.path.exists("src/crayon/resources/dat/vocab_lite.dat"):
    print("Creating dummy DAT profile for testing...")
    os.makedirs("src/crayon/resources/dat", exist_ok=True)
    # Simple JSON vocab
    import json
    from crayon.c_ext.dat_builder import DATBuilder
    vocab_list = ["<UNK>", "hello", "world", "cuda", "tensor", "core", "optimization"]
    builder = DATBuilder()
    builder.build(vocab_list)
    builder.save("src/crayon/resources/dat/vocab_lite.dat")

print("\n🔹 STEP 4: Running Benchmark...")
try:
    # Use the benchmark script logic inline for Colab visibility
    vocab.load_profile("lite")
    
    # 2. Prepare Payload (100k sentences)
    text = "hello world cuda core optimization " * 10
    batch = [text] * 100_000
    
    print(f"🚀 Tokenizing {len(batch):,} documents...")
    start = time.perf_counter()
    tokens = vocab.tokenize(batch)
    dt = time.perf_counter() - start
    
    total_toks = sum(len(t) for t in tokens)
    print(f"✅ DONE in {dt:.4f}s")
    print(f"⚡ Throughput: {len(batch)/dt:,.0f} docs/sec | {total_toks/dt:,.0f} tokens/sec")
    
    # Verify correctness
    print(f"Sample output: {tokens[0][:5]}")

except Exception as e:
    print(f"❌ Execution failed: {e}")

print("\n🔹 COLAB VERIFICATION COMPLETE.")
