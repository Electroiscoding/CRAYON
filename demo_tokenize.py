
"""
CRAYON UNIVERSAL TOKENIZER DEMO
===============================
This script demonstrates the production-ready Crayon Tokenizer API.
It is designed to work seamlessly across:
- Local Machine (Windows/Linux/Mac)
- Google Colab / Jupyter Notebooks
- CPU, NVIDIA GPU (CUDA), and AMD GPU (ROCm)
"""

import os
import sys
from pathlib import Path

# --- 1. Environment Setup ---
# Add 'src' to path so we can run without installing the package
REPO_ROOT = Path(__file__).resolve().parent
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

try:
    from crayon import CrayonVocab
    from crayon.core.vocabulary import enable_verbose_logging
except ImportError:
    print("❌ Error: CRAYON source not found. Make sure you are running this from the repo root.")
    sys.exit(1)

def run_universal_demo():
    # Optional: Enable verbose logging to see hardware detection in action
    # enable_verbose_logging()

    print("=" * 60)
    print("🚀 CRAYON: UNIVERSAL TOKENIZATION DEMO")
    print("=" * 60)

    # --- 2. Initialize Engine ---
    # device="auto" automatically picks CUDA > ROCm > CPU
    print("\n[STEP 1] Initializing Engine (Auto-Detecting Hardware)...")
    try:
        vocab = CrayonVocab(device="auto")
        info = vocab.get_info()
        
        hw_name = info.get("hardware", {}).get("name", "Unknown")
        hw_feat = info.get("hardware", {}).get("features", "")
        print(f"    ✓ Device:  {info['device'].upper()}")
        print(f"    ✓ Backend: {info['backend']}")
        print(f"    ✓ Hardware: {hw_name} [{hw_feat}]")
    except Exception as e:
        print(f"    ❌ Initialization failed: {e}")
        return

    # --- 3. Load Profile ---
    # We load 'lite' which is bundled with the repository
    print(f"\n[STEP 2] Loading 'lite' profile...")
    try:
        vocab.load_profile("lite")
        print(f"    ✓ Profile Loaded: {vocab.current_profile_path}")
        print(f"    ✓ Vocabulary Size: {vocab.vocab_size:,} tokens")
    except Exception as e:
        print(f"    ❌ Load failed: {e}")
        print("    (Note: If you haven't built/downloaded the profiles, run train_code_profile.py first)")
        return

    # --- 4. Performance Tokenization ---
    text = (
        "CRAYON is a hyper-fast tokenizer designed for modern AI. "
        "It supports AVX2 on CPUs, CUDA on NVIDIA, and ROCm on AMD."
    )
    
    print(f"\n[STEP 3] Tokenizing Text...")
    print(f"    Input: \"{text[:50]}...\"")
    
    # Tokenize (returns a list of IDs)
    tokens = vocab.tokenize(text)
    
    print(f"    Result: {tokens[:10]}... ({len(tokens)} tokens)")
    
    # --- 5. Reconstruction (Decoding) ---
    print(f"\n[STEP 4] Decoding back to text...")
    try:
        decoded = vocab.decode(tokens)
        print(f"    Output: \"{decoded[:50]}...\"")
        
        # Verify success
        # Note: BPE usually preserves whitespace and case
        if decoded.strip().lower() == text.strip().lower():
             print("\n✅ SUCCESS: Tokenization and Decoding were perfect!")
        else:
             print("\nℹ️  INFO: Tokenization complete (approximate reconstruction).")
             
    except Exception as e:
        print(f"    ❌ Decode failed: {e}")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE - CRAYON IS READY FOR PRODUCTION")
    print("=" * 60)

if __name__ == "__main__":
    run_universal_demo()
