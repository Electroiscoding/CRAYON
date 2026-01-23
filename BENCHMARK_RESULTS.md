# XERV Crayon V2.0 - Competitive Benchmark Results

**100% HONEST. NO SUGARCOATING. DATA-DRIVEN.**

**Date:** 2026-01-23 16:18:28

**Test Text Size:** 30,800 bytes (30.1 KB)

**Iterations:** 10 (+ 2 warmup)

---

## Results (Real Tokenizers Only - Sorted by Speed)

| Tokenizer | Vocab Size | Token Count | Tokens/sec | MB/sec | Load Time | Avg Time | Min Time | Max Time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CRAYON (lite, 50k)** | 50,000 | 22,100 | 13,957,307 | 18.55 | 4.05ms | 1.58ms | 1.04ms | 2.00ms |
| **tiktoken (cl100k/GPT-4)** | 100,000 | 9,000 | 283,162 | 0.92 | 0.01ms | 31.78ms | 20.92ms | 46.89ms |
| **HF T5 (SentencePiece)** | 32,000 | 12,601 | 280,173 | 0.65 | 1887.32ms | 44.98ms | 35.06ms | 55.59ms |
| **tiktoken (p50k/GPT-3)** | 50,000 | 11,900 | 279,852 | 0.69 | 0.01ms | 42.52ms | 32.64ms | 53.30ms |
| **HF LLaMA (SP-BPE)** | 32,000 | 11,401 | 243,621 | 0.63 | 1503.70ms | 46.80ms | 29.94ms | 61.36ms |
| **HF GPT-2 (BPE)** | 50,257 | 15,700 | 204,496 | 0.38 | 2015.81ms | 76.77ms | 61.45ms | 94.46ms |
| **HF BERT (WordPiece)** | 30,522 | 11,402 | 142,717 | 0.37 | 2588.96ms | 79.89ms | 57.49ms | 192.06ms |

---

## Visualization

![Benchmark Comparison](benchmark_comparison.png)

---

## Speed Comparison

| Tokenizer | Speed vs CRAYON |
| :--- | ---: |
| **CRAYON (lite, 50k)** | **baseline** |
| tiktoken (cl100k/GPT-4) | 49.3x slower |
| HF T5 (SentencePiece) | 49.8x slower |
| tiktoken (p50k/GPT-3) | 49.9x slower |
| HF LLaMA (SP-BPE) | 57.3x slower |
| HF GPT-2 (BPE) | 68.3x slower |
| HF BERT (WordPiece) | 97.8x slower |

---

## Tokenizers Tested

| Tokenizer | Type | Vocab Size | Source |
| :--- | :--- | ---: | :--- |
| CRAYON (lite) | DAT + C++ | 50,000 | Custom engine |
| tiktoken cl100k | BPE | 100,000 | OpenAI GPT-4 |
| tiktoken p50k | BPE | 50,000 | OpenAI GPT-3 |
| HF GPT-2 | BPE (Rust) | 50,257 | HuggingFace |
| HF BERT | WordPiece | 30,522 | HuggingFace |
| HF T5 | SentencePiece | 32,000 | HuggingFace |

---

## Reproducibility

```bash
pip install tiktoken transformers matplotlib
python benchmark_competitive.py
```
