# 🖍️ XERV Crayon

**The Hyper-Production, Cartridge-Based Tokenizer for Specialized AI.**

[![PyPI version](https://badge.fury.io/py/xerv-crayon.svg)](https://badge.fury.io/py/xerv-crayon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![AVX2](https://img.shields.io/badge/SIMD-AVX2-green.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
[![Build Status](https://github.com/xerv/crayon/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/xerv/crayon/actions)

**Crayon** is a next-generation tokenizer designed for **specialization**. Instead of forcing a single, bloated vocabulary on every problem, Crayon uses a **"Cartridge System"** that allows you to hot-swap vocabulary profiles optimized for your specific domain—whether it's Quantum Physics, Rust Programming, or Financial Law.

---

## 🚀 Key Features

*   **💾 The Cartridge System**: Instantly load specialized vocabularies ("Cartridges") like `science`, `code`, or `multilingual`.
*   **⚡ AVX2 Double-Array Trie Engine**: Validated throughput of **~10 Million tokens/sec** using SIMD-accelerated branchless tokenization.
*   **🗺️ Zero-Copy Memory Mapping**: DAT files are memory-mapped directly for instant load times and minimal RAM usage.
*   **🌊 Zero-Disk Streaming**: Builds profiles by streaming data directly from Hugging Face—no multi-gigabyte dataset downloads required.
*   **🛡️ Local Resilience**: Seamlessly falls back to local bootstrap corpuses if internet access is unavailable. Works offline out-of-the-box.
*   **🧠 Entropy-Guided Construction**: Uses information-theoretic principles to select the most "valuable" tokens for a given domain.

---

## 🏎️ DAT Engine V2 Architecture

Crayon V2 uses a **God Tier** implementation combining:

1. **Python Offline Compiler** (`dat_builder.py`): First-Fit algorithm to pack vocabularies into compact Double-Array Trie binary format.
2. **C++ AVX2 Runtime** (`engine.cpp`): Branchless state transitions with SIMD parallel ASCII detection.
3. **Zero-Copy Memory Mapping**: DAT files are loaded via `mmap` with Python buffer protocol for instant startup.

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────────┐
│ vocab.json  │ ──▶  │ DATBuilder   │ ──▶  │  vocab.dat  │ ──▶  │  C++ Engine  │
│   (List)    │      │  (Python)    │      │  (Binary)   │      │   (AVX2)     │
└─────────────┘      └──────────────┘      └─────────────┘      └──────────────┘
```

---

## 📦 Installation

```bash
git clone https://github.com/Xerv-AI/crayon.git
cd crayon
pip install -e .
```

### Build the AVX2 Extension

```bash
python setup.py build_ext --inplace
```

*Note: Requires a C++ compiler (MSVC on Windows, GCC/Clang on Linux/Mac).*

---

## ⚡ Quick Start

### Option 1: Direct DAT Compilation (Fastest to Get Started)

```python
import json
import mmap
from crayon.c_ext.dat_builder import DATBuilder
from crayon.c_ext import crayon_fast

# Load any trained vocabulary (these are in the project root)
with open("trained_vocab_code.json", "r") as f:
    vocab_list = json.load(f)

# Compile to DAT (one-time, takes a few seconds for small vocabs)
builder = DATBuilder()
builder.build(vocab_list)
builder.save("vocab_code.dat")

# Load into C++ engine via memory mapping (instant, <1ms)
with open("vocab_code.dat", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    crayon_fast.load_dat(mm)

# Ultra-fast tokenization (10M+ tokens/sec)
code_snippet = "fn main() { println!(\"Hello, World!\"); }"
tokens = crayon_fast.tokenize(code_snippet)
print(f"Tokens: {tokens}")
```

### Option 2: Using Profile System (Requires Cached Profiles)

```python
from crayon.core.vocabulary import CrayonVocab

# If you've run compile_profiles.py and have cached .dat files:
vocab = CrayonVocab.load_profile("code")
tokens = vocab.tokenize("fn main() { }")
decoded = vocab.decode(tokens)
print(f"Decoded: {decoded}")
```

### 🔧 One-Time Setup: Compile All Profiles

To use the convenient `load_profile()` API, compile profiles once:

```bash
# This builds .dat files and caches them to ~/.cache/xerv/crayon/profiles/
python compile_profiles.py
```

**This is a one-time operation** (or whenever vocabularies are updated). Each profile compilation takes 38ms-26s depending on size. See [DAT_BUILDING_EXPLAINED.md](DAT_BUILDING_EXPLAINED.md) for details.


### 🚀 Try the Demo

Run the included demo script to verifying everything works:

```bash
python demo_tokenize.py
```
*Expected Output:*
```
[1] Loading 'lite' profile...
    Status: 🚀 Fast C++ DAT Engine
[2] Tokenizing: 'Hello, world! This is Crayon.'
    Tokens IDs: [...]
```

---





## 📊 Competitive Benchmarks

**100% HONEST. NO SUGARCOATING. DATA-DRIVEN.**

Run `python benchmark_competitive.py` to reproduce these results yourself.

### Test Environment
- Windows AMD64, Python 3.13.1
- Test Text: 68.4 KB mixed content (code, prose, multilingual)
- 10 iterations + 2 warmup per tokenizer
- Full methodology: [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)

### Results (Real Tokenizers Only - Sorted by Speed)

| Tokenizer | Vocab Size | Tokens/sec | MB/sec | Load Time | Avg Time |
| :--- | ---: | ---: | ---: | ---: | ---: |
| **CRAYON (lite, 50k)** | 50,000 | **6,010,525** | **15.33** | **0.54ms** | **4.56ms** |
| tiktoken (cl100k/GPT-4) | 100,000 | 524,469 | 2.18 | 0.01ms | 32.03ms |
| tiktoken (p50k/GPT-3) | 50,000 | 466,823 | 1.55 | 0.00ms | 44.98ms |
| HF LLaMA (SP-BPE) | 32,000 | 281,558 | 0.95 | 1212.02ms | 73.52ms |
| HF GPT-2 (BPE) | 50,257 | 237,117 | 0.69 | 2051.18ms | 100.79ms |
| HF BERT (WordPiece) | 30,522 | 202,269 | 0.73 | 1603.10ms | 95.43ms |
| HF T5 (SentencePiece) | 32,000 | 189,928 | 0.68 | 1727.91ms | 102.15ms |

### Speed Comparison vs CRAYON

| Tokenizer | Speed vs CRAYON |
| :--- | ---: |
| **CRAYON (lite, 50k)** | **baseline** |
| tiktoken (cl100k/GPT-4) | 11.5x slower |
| tiktoken (p50k/GPT-3) | 12.9x slower |
| HF LLaMA (SP-BPE) | 21.3x slower |
| HF GPT-2 (BPE) | 25.3x slower |
| HF BERT (WordPiece) | 29.7x slower |
| HF T5 (SentencePiece) | 31.6x slower |

### Key Findings (Honest Assessment)

✅ **CRAYON is 11.5x faster than tiktoken** (GPT-4's tokenizer)  
✅ **CRAYON is 25x faster than HuggingFace GPT-2**  
✅ **CRAYON load time is 0.54ms** vs 1-2 seconds for HuggingFace  

### Visualization

![Benchmark Comparison](benchmark_comparison.png)

---


## 🧩 Available Cartridges

Crayon comes with 5 production-ready profiles defined in `src/crayon/core/profiles.py`:

| Profile | Size | Optimized For | Sources |
| :--- | :--- | :--- | :--- |
| **`lite`** | 50k | **Speed & Mobile**. General English and basic logic. | WikiText, RainDrop |
| **`science`** | 250k | **Reasoning**. LaTeX, Quantum Physics, Graduate Math. | GRAD, Physics-700 |
| **`code`** | 250k | **Syntax**. Python, Rust, C++, JavaScript. | CodeParrot, The Stack |
| **`multilingual`** | 250k | **Global**. European languages, Chinese, Hindi. | OSCAR, Wikipedia |
| **`arts_commerce`** | 250k | **Business**. Legal contracts, Financial reports, Literature. | PG19, Financial Phrasebank |

To load any of these:

```python
vocab = CrayonVocab.load_profile("science")
vocab = CrayonVocab.load_profile("multilingual")
```

---

## 🛠️ Advanced Usage

### Compile Vocabulary to DAT Format

```python
from crayon.c_ext.dat_builder import DATBuilder
import json

# Load vocabulary
with open("trained_vocab_lite.json", "r") as f:
    vocab = json.load(f)

# Compile to DAT
builder = DATBuilder()
builder.build(vocab)
builder.save("vocab_lite.dat")
```

### Direct C++ Engine Access

```python
import mmap
from crayon.c_ext import crayon_fast

# Zero-copy load via mmap
with open("vocab_lite.dat", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    crayon_fast.load_dat(mm)

# Ultra-fast tokenization
tokens = crayon_fast.tokenize("Your text here")
```

### Force Rebuild / Offline Mode

```python
# Force a rebuild from local resources only (Fastest)
vocab = CrayonVocab.load_profile("arts_commerce", force_rebuild=True)
```

---

## 🏗️ Architecture

Crayon's architecture is split into four layers:

1.  **Builder (`c_ext/dat_builder.py`)**: Offline compiler that packs vocabularies into DAT binary format.
2.  **Engine (`c_ext/engine.cpp`)**: AVX2 SIMD runtime with branchless tokenization.
3.  **Configuration (`core/profiles.py`)**: The "Menu" of available cartridges.
4.  **Resources (`resources.py`)**: The "Factory" that streams data, manages local fallbacks, and handles atomic caching.

### Key Files

| File | Purpose |
| :--- | :--- |
| `src/crayon/c_ext/dat_builder.py` | Python DAT compiler with First-Fit algorithm |
| `src/crayon/c_ext/engine.cpp` | C++ AVX2 branchless tokenizer |
| `src/crayon/core/vocabulary.py` | High-level Python interface |
| `setup.py` | Build configuration with AVX2 flags |

For a deep dive into the engineering principles behind Crayon, read our [Engineering Treatise](src/crayon/resources/engineering_treatise.md).

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run DAT engine tests specifically
python -m pytest tests/test_c_ext.py -v
```

All 14 DAT engine tests pass:
- `TestDATBuilder` - Compiler tests
- `TestCrayonFastModule` - C++ module tests
- `TestCrayonVocabIntegration` - Full pipeline tests
- `TestVocabularyFallback` - Python fallback tests

---

## 🔬 Verification

To verify the DAT engine is working correctly:

```bash
python verify_dat_engine.py
```

Expected output:
```
============================================================
XERV CRAYON V2.0 - HYPER-PRODUCTION DAT ENGINE VERIFICATION
============================================================
Vocabulary Size: 50,000 tokens
DAT Nodes: 163,000+
Throughput: 9,786,707 tokens/sec
STATUS: ✅ HYPER-PRODUCTION READY
```

---

## 📊 Performance & Training Report

### 🟢 Official Benchmarks (2026-01-20)

| Metric | Result | Notes |
|:---|:---|:---|
| **Best Throughput** | 10,416,668 tokens/sec | science profile |
| **50k Vocab Throughput** | 5,082,050 tokens/sec | lite profile |
| **Load Time** | 0.02ms | All profiles |
| **Engine** | DAT V2 + Buffer Protocol | Production |
| **Tests** | 14/14 Passed | Verified |

*See [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md) for complete methodology.*


### 💾 Exact Training Data Used

The default "batteries included" vocabulary was constructed using the following specific quantities of high-entropy text:

| Dataset | Size | Samples | Description |
|:---|:---|:---|:---|
| **Tiny Shakespeare** | 1.06 MB | 1 (Full) | Classical Literature |
| **RainDrop-DTS** | 179 KB | 3,210 | Instruction Following |
| **Physics** | 332 KB | 700 | Scientific Reasoning |
| **GRAD Math** | 5.00 MB | 500* | Graduate Mathematics |
| **TOTAL** | **~6.56 MB** | **4,411** | **Curated Corpus** |

*GRAD dataset limited to 500 high-density samples for efficient default build.*

---

## 🧩 API Reference

### CrayonVocab

```python
# Constructors
CrayonVocab(tokens: List[str], unk_token: str = "<UNK>")
CrayonVocab.from_corpus(corpus: str, target_size: int = 500000)
CrayonVocab.from_default_sources(vocab_size: int = 500000)
CrayonVocab.from_file(path: str)
CrayonVocab.from_json(path: str)
CrayonVocab.load_profile(name: str)  # NEW: Load cached DAT profiles

# Methods
vocab.tokenize(text: str) -> List[int]  # Uses C++ engine if available
vocab.decode(token_ids: List[int]) -> str
vocab.save(path: str, format: str = "txt")
```

### DAT Builder

```python
from crayon.c_ext.dat_builder import DATBuilder

builder = DATBuilder()
builder.build(vocab_list: List[str])  # Compile to DAT
builder.save(output_path: str)        # Save binary file
```

### C++ Engine

```python
from crayon.c_ext import crayon_fast

crayon_fast.load_dat(buffer)  # Load from bytes, mmap, or memoryview
crayon_fast.tokenize(text: str) -> List[int]  # Ultra-fast tokenization
```

### Utilities

```python
from crayon import check_c_extension, check_resources

# Check if SIMD-accelerated C extension is available
print(check_c_extension())  # True/False

# Check available data sources
print(check_resources())
```

---

## 📜 Citation

If you use Crayon in your research, please cite:

```bibtex
@techreport{xerv2026crayon,
  title={XERV Crayon: A First-Principles Analysis of Production-Grade Tokenization},
  author={Pal, Soham and Xerv Research},
  year={2026},
  institution={Xerv Research Engineering Division}
}
```

---

## 📄 License

Copyright (c) 2025-2026 Xerv Research. Released under the MIT License.

---

**Built with 💙 by Xerv Research Engineering Division.**