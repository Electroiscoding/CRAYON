"""
Test all code examples from README.md to ensure they work correctly.
"""
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), "build", "lib.win-amd64-cpython-313"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

print("=" * 70)
print("TESTING README CODE EXAMPLES")
print("=" * 70)
print()

# Test 1: Quick Start Example
print("[TEST 1] Quick Start - Load Profile and Tokenize")
print("-" * 70)
try:
    from crayon.core.vocabulary import CrayonVocab
    
    vocab = CrayonVocab(device="auto")
    vocab.load_profile("lite")
    
    # Tokenize specialized syntax
    code_snippet = "fn main() { println!(\"Hello, World!\"); }"
    tokens = vocab.tokenize(code_snippet)
    
    # Check if decode works
    try:
        decoded = vocab.decode(tokens)
        print(f"✓ Tokenize: {code_snippet}")
        print(f"✓ Tokens: {tokens}")
        print(f"✓ Decoded: {decoded}")
        print("✓ TEST PASSED")
    except AttributeError:
        print(f"⚠ WARNING: vocab.decode() not implemented yet")
        print(f"✓ Tokenize works: {tokens}")
        print("✓ TEST PARTIALLY PASSED")
except Exception as e:
    print(f"✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 2: Load different profiles
print("[TEST 2] Load Different Profiles")
print("-" * 70)
for profile_name in ["lite", "standard"]:
    try:
        vocab = CrayonVocab(device="auto")
        vocab.load_profile(profile_name)
        print(f"✓ Loaded '{profile_name}' profile")
    except Exception as e:
        print(f"✗ Failed to load '{profile_name}': {e}")

print()

# Test 3: DAT Builder Example
print("[TEST 3] Compile Vocabulary to DAT Format")
print("-" * 70)
try:
    from crayon.c_ext.dat_builder import DATBuilder
    import json
    import tempfile
    
    # Use a small test vocab
    test_vocab = ["hello", "world", "test", "python"]
    
    # Compile to DAT
    builder = DATBuilder()
    builder.build(test_vocab)
    
    # Save to temp file
    dat_path = os.path.join(tempfile.gettempdir(), "test_readme.dat")
    builder.save(dat_path)
    
    print(f"✓ Built DAT with {builder.size} nodes")
    print(f"✓ Saved to {dat_path}")
    
    os.unlink(dat_path)
    print("✓ TEST PASSED")
except Exception as e:
    print(f"✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

print()

# Test 4: Direct C++ Engine Access
print("[TEST 4] Direct C++ Engine Access")
print("-" * 70)
try:
    import mmap
    from crayon.c_ext import crayon_fast
    from crayon.c_ext.dat_builder import DATBuilder
    import tempfile
    
    # Build a small DAT
    test_vocab = ["the", "quick", "brown", "fox"]
    builder = DATBuilder()
    builder.build(test_vocab)
    
    dat_path = os.path.join(tempfile.gettempdir(), "test_engine.dat")
    builder.save(dat_path)
    
    # Zero-copy load via mmap
    with open(dat_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        size = crayon_fast.load_dat(mm)
    
    # Ultra-fast tokenization
    tokens = crayon_fast.tokenize("the quick brown fox")
    
    print(f"✓ Loaded DAT: {size} nodes")
    print(f"✓ Tokenized: {tokens}")
    
    os.unlink(dat_path)
    print("✓ TEST PASSED")
except Exception as e:
    print(f"✗ TEST FAILED: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("README CODE TESTS COMPLETE")
print("=" * 70)
