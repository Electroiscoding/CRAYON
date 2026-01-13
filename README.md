# XERV Crayon 🖍️

[![PyPI version](https://badge.fury.io/py/xerv-crayon.svg)](https://badge.fury.io/py/xerv-crayon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/xerv/crayon/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/xerv/crayon/actions)

**Crayon** is a production-grade tokenizer achieving unprecedented performance (>2M tokens/s) through rigorous first-principles engineering. It implements a hybrid Trie/Hash architecture with AVX2 SIMD optimizations and zero-copy memory management.

> "A Complete Engineering Treatise on Ultra-High-Throughput Text Processing" - Xerv Research

## 🚀 Key Features

- **Extreme Throughput:** >2,100,000 tokens/second on standard hardware
- **Zero-Copy Architecture:** Memory-mapped file processing for datasets larger than RAM
- **Hardware-Aligned:** Cache-aware `TrieNode` structures (64-byte aligned) and AVX2 vectorization
- **Adaptive Vocabulary:** Entropy-guided vocabulary evolution for out-of-distribution text
- **Multilingual Native:** SIMD-accelerated Unicode NFC normalization
- **Batteries Included:** Built-in vocabulary from curated datasets (no setup required)

## 📦 Installation

Crayon requires a C99-compliant compiler and a CPU with AVX2 support (most modern Intel/AMD chips).

```bash
# Basic installation
pip install xerv-crayon

# With streaming vocabulary builder (HuggingFace datasets)
pip install xerv-crayon[full]
```

### Building from Source

```bash
git clone https://github.com/xerv/crayon.git
cd crayon
pip install -e .
```

## ⚡ Quick Start

### Option 1: Load Existing Vocabulary

```python
from crayon import CrayonVocab

# Initialize with your vocabulary
vocab = CrayonVocab(["hello", "world", "!", "<UNK>"])

# High-speed tokenization (Uses C-Extension if available)
tokens = vocab.tokenize("hello world!")
print(tokens)  # Output: [0, 1, 2]
```

### Option 2: Build from Your Corpus

```python
from crayon import CrayonVocab

# Train vocabulary from your text
vocab = CrayonVocab.from_corpus(
    "Your training text here. Add more text for better coverage.",
    target_size=50000
)

tokens = vocab.tokenize("Your text to tokenize")
```

### Option 3: Batteries-Included (Recommended)

```python
from crayon import CrayonVocab

# Build vocabulary from curated sources (GRAD, Physics, Shakespeare, etc.)
# Streams data directly - no local files needed!
vocab = CrayonVocab.from_default_sources(vocab_size=50000)

tokens = vocab.tokenize("Hello world!")
decoded = vocab.decode(tokens)
```

### Streaming Large Files (Zero-Copy)

```python
from crayon import ZeroCopyTokenizer, CrayonVocab

vocab = CrayonVocab.from_file("vocab.txt")
zc_tokenizer = ZeroCopyTokenizer(vocab)

# Process 100GB file with <10MB RAM usage
for token_id, offset in zc_tokenizer.tokenize_file_zerocopy("huge_corpus.txt"):
    process_token(token_id)
```

### Pipeline Parallelization

```python
from crayon import PipelineTokenizer, CrayonVocab

vocab = CrayonVocab(["hello", "world", " "])
pipeline = PipelineTokenizer(vocab)
pipeline.start_pipeline()

# Submit texts asynchronously
pipeline.submit_text("doc_1", "hello world")
pipeline.submit_text("doc_2", "world hello")

# Retrieve results
result1 = pipeline.get_result()
result2 = pipeline.get_result()

pipeline.stop_pipeline()
```

## 🛠️ Architecture

Crayon solves the quadratic complexity of BPE using a three-tiered optimization strategy:

1. **Algorithmic:** O(1) expected lookup time via cache-aligned Trie with SIMD search
2. **Memory:** `__slots__` optimized metadata and buffer pooling
3. **Hardware:** AVX2 SIMD instructions for parallel character comparison

### Data Structures

- **64-byte aligned TrieNode:** Fits exactly one CPU cache line
- **SIMD child lookup:** 16-way parallel character search using SSE2
- **Bitmap existence check:** O(1) ASCII child detection

## 🧪 Benchmarks

| Tokenizer | Throughput (tok/s) | Memory Peak (MB) |
|-----------|-------------------|------------------|
| **Crayon** | **2,100,000** | **128** |
| SentencePiece | 850,000 | 245 |
| WordPiece | 620,000 | 198 |

*Benchmarked on AMD Ryzen 9 7950X*

### Run Benchmarks

```bash
python benchmarks/run_benchmarks.py
```

## 🧩 API Reference

### CrayonVocab

```python
# Constructors
CrayonVocab(tokens: List[str], unk_token: str = "<UNK>")
CrayonVocab.from_corpus(corpus: str, target_size: int = 500000)
CrayonVocab.from_default_sources(vocab_size: int = 500000)
CrayonVocab.from_file(path: str)
CrayonVocab.from_json(path: str)

# Methods
vocab.tokenize(text: str) -> List[int]
vocab.decode(token_ids: List[int]) -> str
vocab.save(path: str, format: str = "txt")
```

### Utilities

```python
from crayon import check_c_extension, check_resources

# Check if SIMD-accelerated C extension is available
print(check_c_extension())  # True/False

# Check available data sources
print(check_resources())
```

## 📜 Citation

If you use Crayon in your research, please cite:

```bibtex
@techreport{xerv2025crayon,
  title={XERV Crayon: A First-Principles Analysis of Production-Grade Tokenization},
  author={Pal, Soham and Xerv Research},
  year={2025},
  institution={Xerv Research Engineering Division}
}
```

## 📄 License

Copyright (c) 2025 Xerv Research. Released under the MIT License.