# DAT Building: One-Time vs Every-Time - Detailed Explanation

## Overview

**DAT (Double-Array Trie) Building** is the process of converting a text-based vocabulary (JSON/list) into an optimized binary format that enables ultra-fast tokenization.

---

## The Building Process

### What Happens During DAT Building?

1. **Trie Construction** (Step 1)
   - Converts each vocabulary token into a tree structure
   - Each character/byte becomes a node in the tree
   - Common prefixes share the same path (e.g., "apple" and "apply" share "appl")

2. **Array Packing** (Step 2 - The Expensive Part)
   - Uses a "First-Fit" algorithm to find optimal positions in integer arrays
   - Compresses the tree into 3 parallel arrays: `base`, `check`, `values`
   - **This is computationally expensive**: O(n×m) where n=vocab_size, m=avg_token_length
   
3. **Binary Serialization** (Step 3)
   - Writes the arrays to a `.dat` binary file
   - Format: `[MAGIC|VERSION|SIZE|BASE_ARRAY|CHECK_ARRAY|VALUES_ARRAY]`
   - Enables memory-mapping for instant zero-copy loading

### Performance Cost

| Vocabulary Size | Build Time | DAT File Size |
|-----------------|------------|---------------|
| 367 tokens | ~38ms | 5 KB |
| 5,000 tokens | ~26s | 143 KB |
| 50,000 tokens | ~5-10min | ~1.5 MB |

---

## One-Time vs Every-Time

### ✅ CORRECT APPROACH: One-Time Build + Cache

**Build Once:**
- Run `compile_profiles.py` during:
  - Package development
  - First-time user setup
  - CI/CD pipeline

**Cache Forever:**
- Save `.dat` files to: `~/.cache/xerv/crayon/profiles/`
- OR distribute pre-built `.dat` files with the package
- Users never rebuild unless vocabulary changes

**Runtime:**
```python
# This should be INSTANT (just mmap)
vocab = CrayonVocab.load_profile("code")  # <1ms to load .dat
tokens = vocab.tokenize(text)              # 10M+ tokens/sec
```

### ❌ INCORRECT APPROACH: Build Every Time

```python
# BAD: Building from JSON every import
builder = DATBuilder()
builder.build(vocab)  # Takes 26 seconds for 5k vocab!
```

This would make the library unusable.

---

## Current Implementation Status

### What Works ✅

1. **DATBuilder** (`src/crayon/c_ext/dat_builder.py`)
   - ✅ Compiles vocab to DAT format
   - ✅ Saves binary files

2. **CrayonVocab.load_profile()** (`src/crayon/core/vocabulary.py`)
   - ✅ Checks for cached `.dat` file first
   - ✅ Falls back to `.json` if `.dat` not found
   - ✅ Calls `build_and_cache_profile()` if neither exists

3. **C++ Engine** (`src/crayon/c_ext/engine.cpp`)
   - ✅ Memory-maps `.dat` files via Python buffer protocol
   - ✅ Zero-copy instant loading (<1ms)
   - ✅ AVX2 SIMD tokenization (10M+ tok/sec)

### What's Missing ⚠️

1. **Pre-built .dat files not distributed**
   - Currently, `.dat` files must be built manually via `compile_profiles.py`
   - Should be included in package or built during `pip install`

2. **Vocabulary files not in cache**
   - `trained_vocab_*.json` files exist in project root
   - Not automatically copied to `~/.cache/xerv/crayon/profiles/`
   - `build_and_cache_profile()` should handle this

3. **`decode()` method missing**
   - README examples show `vocab.decode(tokens)`
   - Method doesn't exist in `CrayonVocab` class

---

## Recommended Workflow

### For Package Developers:

```bash
# 1. Train vocabularies (already done - trained_vocab_*.json exist)
python train_vocab.py

# 2. Compile to DAT format
python compile_profiles.py

# 3. Distribute .dat files with package
# - Include in MANIFEST.in
# - Copy to package installation directory
```

### For End Users:

```python
# Should just work (instant load from cached .dat)
from crayon import CrayonVocab
vocab = CrayonVocab.load_profile("code")  # <1ms
```

---

## Summary

| Aspect | Answer |
|--------|--------|
| **One-time or Every-time?** | **ONE-TIME** per vocabulary version |
| **Who builds?** | Developer OR first-time user setup |
| **Build frequency?** | Only when vocabulary changes |
| **Runtime cost?** | **<1ms** (just mmap, no rebuild) |
| **User experience?** | Instant, zero compilation delay |

**The DAT file is like a compiled binary** - you compile your source code once, then distribute/cache the binary for instant execution.
