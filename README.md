<p align="center">
  <img src="https://em-content.zobj.net/source/microsoft-teams/363/crayon_1f58d-fe0f.png" width="120" alt="Crayon Logo"/>
</p>

<h1 align="center">🖍️ XERV Crayon</h1>

<p align="center">
  <strong>The Cartridge-Based Tokenizer for Specialized AI</strong>
</p>

<p align="center">
  <a href="https://badge.fury.io/py/xerv-crayon"><img src="https://badge.fury.io/py/xerv-crayon.svg" alt="PyPI version"/></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"/></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.12+-blue.svg" alt="Python 3.12+"/></a>
  <a href="https://en.wikipedia.org/wiki/Advanced_Vector_Extensions"><img src="https://img.shields.io/badge/SIMD-AVX2-green.svg" alt="AVX2"/></a>
  <a href="https://github.com/xerv/crayon/actions"><img src="https://github.com/xerv/crayon/actions/workflows/build_wheels.yml/badge.svg" alt="Build Status"/></a>
</p>

<p align="center">
  <em>Why force a single bloated vocabulary on every problem?</em><br/>
  <strong>Crayon</strong> is a next-generation tokenizer designed for <strong>specialization</strong>. Hot-swap vocabulary profiles ("Cartridges") optimized for your domain—Quantum Physics, Rust Programming, Financial Law, or anything in between.
</p>

---

## 🚀 Key Features

| Feature | Description |
|:--------|:------------|
| **💾 Cartridge System** | Instantly hot-swap specialized vocabularies (`science`, `code`, `multilingual`) |
| **⚡ AVX2 Double-Array Trie** | Validated **~10M tokens/sec** via SIMD-accelerated branchless tokenization |
| **🗺️ Zero-Copy Memory Mapping** | DAT files loaded via `mmap` for instant startup & minimal RAM |
| **🌊 Zero-Disk Streaming** | Build profiles directly from Hugging Face—no multi-GB downloads |
| **🛡️ Offline Resilience** | Seamless local bootstrap fallback. Works offline out-of-the-box |
| **🧠 Entropy-Guided Construction** | Information-theoretic token selection for maximum domain efficiency |

---

## 📊 Benchmarks — The Numbers Speak

> **100% HONEST. NO SUGARCOATING. DATA-DRIVEN.**
> 
> Run `python benchmark_competitive.py` to reproduce these results yourself.

### ⚡ Speed Comparison

| Tokenizer | Tokens/sec | vs CRAYON |
|:----------|----------:|:----------|
| **🖍️ CRAYON (lite, 50k)** | **6,010,525** | **baseline** |
| tiktoken (GPT-4) | 524,469 | 11.5x slower |
| tiktoken (GPT-3) | 466,823 | 12.9x slower |
| HF LLaMA (SP-BPE) | 281,558 | 21.3x slower |
| HF GPT-2 (BPE) | 237,117 | 25.3x slower |
| HF BERT (WordPiece) | 202,269 | 29.7x slower |
| HF T5 (SentencePiece) | 189,928 | 31.6x slower |

### 📈 Full Benchmark Results

| Tokenizer | Vocab Size | Tokens/sec | MB/sec | Load Time | Avg Time |
|:----------|----------:|-----------:|-------:|----------:|---------:|
| **CRAYON (lite, 50k)** | 50,000 | **6,010,525** | **15.33** | **0.54ms** | **4.56ms** |
| tiktoken (cl100k/GPT-4) | 100,000 | 524,469 | 2.18 | 0.01ms | 32.03ms |
| tiktoken (p50k/GPT-3) | 50,000 | 466,823 | 1.55 | 0.00ms | 44.98ms |
| HF LLaMA (SP-BPE) | 32,000 | 281,558 | 0.95 | 1212.02ms | 73.52ms |
| HF GPT-2 (BPE) | 50,257 | 237,117 | 0.69 | 2051.18ms | 100.79ms |
| HF BERT (WordPiece) | 30,522 | 202,269 | 0.73 | 1603.10ms | 95.43ms |
| HF T5 (SentencePiece) | 32,000 | 189,928 | 0.68 | 1727.91ms | 102.15ms |

<details>
<summary><strong>📋 Test Environment & Methodology</strong></summary>

- **Platform:** Windows AMD64, Python 3.13.1
- **Test Text:** 68.4 KB mixed content (code, prose, multilingual)
- **Iterations:** 10 runs + 2 warmup per tokenizer
- **Full methodology:** [BENCHMARK_RESULTS.md](BENCHMARK_RESULTS.md)

</details>

### 🏆 Key Takeaways

| Metric | Result |
|:-------|:-------|
| ✅ vs tiktoken (GPT-4) | **11.5x faster** |
| ✅ vs HuggingFace GPT-2 | **25x faster** |
| ✅ Load time | **0.54ms** (vs 1-2s for HuggingFace) |
| ✅ Peak throughput | **10.4M tokens/sec** (science profile) |

![Benchmark Comparison](benchmark_comparison.png)

---

## ⚡ Quick Start

Get tokenizing in under 60 seconds:

### Option 1: Direct DAT Compilation

```python
import json
import mmap
from crayon.c_ext.dat_builder import DATBuilder
from crayon.c_ext import crayon_fast

# Load any trained vocabulary
with open("trained_vocab_code.json", "r") as f:
    vocab_list = json.load(f)

# Compile to DAT (one-time, few seconds)
builder = DATBuilder()
builder.build(vocab_list)
builder.save("vocab_code.dat")

# Load into C++ engine via memory mapping (instant, <1ms)
with open("vocab_code.dat", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    crayon_fast.load_dat(mm)

# Ultra-fast tokenization 🚀
code = 'fn main() { println!("Hello, World!"); }'
tokens = crayon_fast.tokenize(code)
print(f"Tokens: {tokens}")
```

### Option 2: Profile System (Recommended)

```python
from crayon.core.vocabulary import CrayonVocab

# Load pre-compiled profile (requires one-time compile_profiles.py)
vocab = CrayonVocab.load_profile("code")
tokens = vocab.tokenize("fn main() { }")
decoded = vocab.decode(tokens)
print(f"Decoded: {decoded}")
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

> **Note:** Requires a C++ compiler (MSVC on Windows, GCC/Clang on Linux/Mac).

### 🔧 One-Time Setup: Compile Profiles

```bash
# Builds .dat files → ~/.cache/xerv/crayon/profiles/
python compile_profiles.py
```

Each profile takes 38ms-26s depending on size. See [DAT_BUILDING_EXPLAINED.md](DAT_BUILDING_EXPLAINED.md) for details.

### 🧪 Verify Installation

```bash
python demo_tokenize.py
```

Expected output:
```
[1] Loading 'lite' profile...
    Status: 🚀 Fast C++ DAT Engine
[2] Tokenizing: 'Hello, world! This is Crayon.'
    Tokens IDs: [...]
```

---

## 🏎️ DAT Engine V2 Architecture

Crayon V2 uses a **"God Tier"** implementation combining:

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐      ┌──────────────┐
│ vocab.json  │ ──▶  │ DATBuilder   │ ──▶  │  vocab.dat  │ ──▶  │  C++ Engine  │
│   (List)    │      │  (Python)    │      │  (Binary)   │      │   (AVX2)     │
└─────────────┘      └──────────────┘      └─────────────┘      └──────────────┘
```

| Component | File | Purpose |
|:----------|:-----|:--------|
| **Offline Compiler** | `dat_builder.py` | First-Fit algorithm → compact DAT binary |
| **AVX2 Runtime** | `engine.cpp` | Branchless state transitions + SIMD parallel ASCII |
| **Zero-Copy Loader** | `mmap` + buffer protocol | Instant startup, minimal RAM |

---

## 🧩 Available Cartridges

5 production-ready profiles defined in `src/crayon/core/profiles.py`:

| Profile | Size | Optimized For | Sources |
|:--------|:-----|:--------------|:--------|
| **`lite`** | 50k | Speed & Mobile | WikiText, RainDrop |
| **`science`** | 250k | Reasoning (LaTeX, Quantum, Grad Math) | GRAD, Physics-700 |
| **`code`** | 250k | Syntax (Python, Rust, C++, JS) | CodeParrot, The Stack |
| **`multilingual`** | 250k | Global (EU langs, Chinese, Hindi) | OSCAR, Wikipedia |
| **`arts_commerce`** | 250k | Business (Legal, Finance, Lit) | PG19, Fin Phrasebank |

```python
vocab = CrayonVocab.load_profile("science")
vocab = CrayonVocab.load_profile("multilingual")
```

---

## 🛠️ Advanced Usage

<details>
<summary><strong>Compile Vocabulary to DAT Format</strong></summary>

```python
from crayon.c_ext.dat_builder import DATBuilder
import json

with open("trained_vocab_lite.json", "r") as f:
    vocab = json.load(f)

builder = DATBuilder()
builder.build(vocab)
builder.save("vocab_lite.dat")
```

</details>

<details>
<summary><strong>Direct C++ Engine Access</strong></summary>

```python
import mmap
from crayon.c_ext import crayon_fast

with open("vocab_lite.dat", "rb") as f:
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    crayon_fast.load_dat(mm)

tokens = crayon_fast.tokenize("Your text here")
```

</details>

<details>
<summary><strong>Force Rebuild / Offline Mode</strong></summary>

```python
# Rebuild from local resources only (fastest)
vocab = CrayonVocab.load_profile("arts_commerce", force_rebuild=True)
```

</details>

---

## 🏗️ Architecture

| Layer | File | Purpose |
|:------|:-----|:--------|
| **Builder** | `c_ext/dat_builder.py` | Offline DAT compiler |
| **Engine** | `c_ext/engine.cpp` | AVX2 SIMD runtime |
| **Config** | `core/profiles.py` | Cartridge definitions |
| **Resources** | `resources.py` | Streaming, fallbacks, caching |

For a deep dive, read the [Engineering Treatise](src/crayon/resources/engineering_treatise.md).

---

## 🧪 Testing

```bash
# All tests
python -m pytest tests/ -v

# DAT engine tests
python -m pytest tests/test_c_ext.py -v
```

**14/14 tests pass:** DATBuilder, C++ module, full pipeline, Python fallback.

### 🔬 DAT Engine Verification

```bash
python verify_dat_engine.py
```

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

## 📊 Training Data

| Dataset | Size | Samples | Domain |
|:--------|:-----|:--------|:-------|
| Tiny Shakespeare | 1.06 MB | 1 (Full) | Classical Literature |
| RainDrop-DTS | 179 KB | 3,210 | Instruction Following |
| Physics | 332 KB | 700 | Scientific Reasoning |
| GRAD Math | 5.00 MB | 500* | Graduate Mathematics |
| **TOTAL** | **~6.56 MB** | **4,411** | **Curated Corpus** |

<sub>*GRAD dataset limited to 500 high-density samples for efficient default build.</sub>

---

## 🧩 API Reference

<details>
<summary><strong>CrayonVocab</strong></summary>

```python
# Constructors
CrayonVocab(tokens: List[str], unk_token: str = "<UNK>")
CrayonVocab.from_corpus(corpus: str, target_size: int = 500000)
CrayonVocab.from_default_sources(vocab_size: int = 500000)
CrayonVocab.from_file(path: str)
CrayonVocab.from_json(path: str)
CrayonVocab.load_profile(name: str)  # Load cached DAT profiles

# Methods
vocab.tokenize(text: str) -> List[int]
vocab.decode(token_ids: List[int]) -> str
vocab.save(path: str, format: str = "txt")
```

</details>

<details>
<summary><strong>DAT Builder</strong></summary>

```python
from crayon.c_ext.dat_builder import DATBuilder

builder = DATBuilder()
builder.build(vocab_list: List[str])
builder.save(output_path: str)
```

</details>

<details>
<summary><strong>C++ Engine</strong></summary>

```python
from crayon.c_ext import crayon_fast

crayon_fast.load_dat(buffer)  # bytes, mmap, or memoryview
crayon_fast.tokenize(text: str) -> List[int]
```

</details>

<details>
<summary><strong>Utilities</strong></summary>

```python
from crayon import check_c_extension, check_resources

print(check_c_extension())  # True/False
print(check_resources())     # Available data sources
```

</details>

---

## 🤝 Contributing

We welcome contributions! Whether it's new cartridges, performance optimizations, or bug fixes—open an issue or submit a PR.

---

## 📜 Citation

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

Copyright (c) 2025-2026 Xerv Research. Released under the **MIT License**.

---

<p align="center">
  <strong>Built with 💙 by Xerv Research Engineering Division</strong>
</p>

<p align="center">
  <sub>⭐ Star this repo if Crayon helps your project!</sub>
</p>