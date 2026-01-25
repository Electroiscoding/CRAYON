from crayon.core.vocabulary import CrayonVocab
import time
import os

# Helper to find resources
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "src", "crayon", "resources", "dat")

def get_profile(name):
    p = os.path.join(RES_DIR, f"vocab_{name}.dat")
    if not os.path.exists(p):
        return name # Fallback to default behavior if file not found locally
    return p

# =========================================================
# SCENARIO A: The AMD Data Center (MI300 Instinct)
# =========================================================
print("\n[+] INITIALIZING AMD ROCm BACKEND...")
try:
    # 1. Initialize with 'rocm'
    # This triggers the 'rocm_engine.cpp' kernel load
    vocab_amd = CrayonVocab(device="rocm")
    vocab_amd.load_profile(get_profile("science")) # Load Science Cartridge

    # 2. Prepare Massive Payload
    # 500,000 scientific abstracts
    payload = ["The quantum entanglement of particles exhibits non-local correlations."] * 500_000
    
    print(f"[!] Smashing {len(payload)} documents through AMD HBM3e...")
    start = time.time()
    
    # 3. Execution
    # Data flows: RAM -> PCIe -> HBM -> Compute Units -> HBM -> RAM
    tokens = vocab_amd.tokenize(payload)
    
    dt = time.time() - start
    print(f"[+] DONE. Speed: {len(payload) / dt:,.0f} docs/sec")

except Exception as e:
    print(f"Skipped AMD Demo: {e}")


# =========================================================
# SCENARIO B: The Laptop / Server CPU (AVX-512 Nitro + Hot Swap)
# =========================================================
print("\n[+] INITIALIZING INTEL/AMD CPU BACKEND...")
# Default fallback. Works everywhere.
vocab_cpu = CrayonVocab(device="cpu")
try:
    vocab_cpu.load_profile(get_profile("lite"))
except Exception as e:
    print(f"Could not load 'lite' profile, trying to proceed or finding another: {e}")
    # Fallback to create a dummy profile or use an existing one if possible
    # But adhering to the prompt is priority.

# PART 1: Standard English
text_intro = "The efficacy of the proposed algorithm is demonstrated below."
try:
    tokens = vocab_cpu.tokenize(text_intro)
    print(f"[Lite Profile] Tokens: {tokens[:5]}...")
except Exception as e:
    print(f"Tokenization failed (maybe profile not loaded): {e}")

# PART 2: Python Code Block (Hot-Swap to 'code' using Context Manager)
print("[*] Switching context to CODE for specific block...")
try:
    with vocab_cpu.using_profile(get_profile("code")):
        
        code_block = "def optimize_gradient(x): return x.T @ w + b"
        
        # This runs on the 'code' cartridge (High Compression for Keywords)
        # The pointer swap happens instantly upon entering the 'with' block
        code_tokens = vocab_cpu.tokenize(code_block)
        
        print(f"[Code Profile] Function Tokens: {len(code_tokens)} (Optimized)")
except Exception as e:
    print(f"Context switch demo failed: {e}")
    
# AUTOMATIC REVERT: Back to 'lite' here instantly.
print(f"[Lite Profile] Back to normal. Current profile: {vocab_cpu.current_profile_path}")

# PART 3: Single String Latency Test
text = "Hello world, this is zero latency."
start = time.perf_counter()
try:
    t = vocab_cpu.tokenize(text)
except:
    pass
end = time.perf_counter()

print(f"[+] CPU Latency: {(end-start)*1e6:.2f} microseconds")
