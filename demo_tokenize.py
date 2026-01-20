"""
Crayon Tokenizer Demo
---------------------
Simple script to demonstrate loading a profile and tokenizing text.
"""
import sys
import os
from pathlib import Path

# Add paths to use local build if running from source
sys.path.insert(0, os.path.join(os.getcwd(), "build", "lib.win-amd64-cpython-313"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from crayon.core.vocabulary import CrayonVocab

def run_demo():
    print("=" * 60)
    print("CRAYON TOKENIZER DEMO")
    print("=" * 60)

    # 1. Load Profile
    profile_name = "lite"
    print(f"\n[1] Loading '{profile_name}' profile...")
    
    try:
        vocab = CrayonVocab.load_profile(profile_name)
    except Exception as e:
        print(f"Standard load failed: {e}")
        # Manual fallback for development environment without installation
        print("    -> Attempting development fallback...")
        dat_path = Path("src/crayon/resources/dat/vocab_lite.dat")
        json_path = Path("src/crayon/resources/dat/vocab_lite.json")
        
        if dat_path.exists():
            vocab = CrayonVocab()
            vocab._load_binary_dat(dat_path)
            if json_path.exists():
                vocab._load_json_mappings(json_path)
        else:
            print("❌ Could not find tokenizer files.")
            sys.exit(1)

    # 2. Check Engine Mode
    mode = "🚀 Fast C++ DAT Engine" if vocab.fast_mode else "🐢 Slow Python Fallback"
    print(f"    Status: {mode}")

    # 3. Tokenize
    text = "Hello, world! This is Crayon."
    print(f"\n[2] Tokenizing: '{text}'")
    
    tokens = vocab.tokenize(text)
    print(f"    Tokens IDs: {tokens}")
    print(f"    Count:      {len(tokens)}")

    # 4. Decode
    print(f"\n[3] Decoding back to text...")
    try:
        decoded = vocab.decode(tokens)
        print(f"    Decoded:    '{decoded}'")
        
        if decoded == text:
            print("    Unknown/Unmapped tokens found (exact match requires full coverage)")
        else:
            print("    (Note: exact reconstruction depends on vocabulary coverage)")
            
    except Exception as e:
        print(f"    Decode failed: {e}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    run_demo()
