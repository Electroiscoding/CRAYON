# XERV Crayon 🖍️

[![PyPI version](https://badge.fury.io/py/xerv-crayon.svg)](https://badge.fury.io/py/xerv-crayon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/xerv/crayon/actions/workflows/build_wheels.yml/badge.svg)](https://github.com/xerv/crayon/actions)

**Crayon** is a production-grade tokenizer achieving unprecedented performance (>2M tokens/s) through rigorous first-principles engineering. It implements a hybrid Trie/Hash architecture with AVX2 SIMD optimizations and zero-copy memory management.

> [cite_start]"A Complete Engineering Treatise on Ultra-High-Throughput Text Processing" - Xerv Research [cite: 1-3].

## 🚀 Key Features

- [cite_start]**Extreme Throughput:** >2,100,000 tokens/second on standard hardware[cite: 9].
- [cite_start]**Zero-Copy Architecture:** Memory-mapped file processing for datasets larger than RAM [cite: 836-845].
- [cite_start]**Hardware-Aligned:** Cache-aware `TrieNode` structures (64-byte aligned) and AVX2 vectorization [cite: 217-230].
- [cite_start]**Adaptive Vocabulary:** Entropy-guided vocabulary evolution for out-of-distribution text[cite: 1095].
- [cite_start]**Multilingual Native:** SIMD-accelerated Unicode NFC normalization[cite: 281].

## 📦 Installation

Crayon requires a C99-compliant compiler and a CPU with AVX2 support (most modern Intel/AMD chips).

```bash
pip install xerv-crayon
```