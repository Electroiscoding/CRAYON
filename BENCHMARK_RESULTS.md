# XERV Crayon V2.0 - Competitive Benchmark Results

**100% HONEST. NO SUGARCOATING. DATA-DRIVEN.**

**Date:** 2026-01-23 15:29:11

**Test Text Size:** 30,800 bytes (30.1 KB)

**Iterations:** 10 (+ 2 warmup)

---

## Results (Real Tokenizers Only - Sorted by Speed)

| Tokenizer | Vocab Size | Token Count | Tokens/sec | MB/sec | Load Time | Avg Time | Min Time | Max Time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CRAYON (lite, 50k)** | 50,000 | 22,100 | 20,293,475 | 26.97 | 2.18ms | 1.09ms | 0.63ms | 1.52ms |
| **tiktoken (p50k/GPT-3)** | 50,000 | 11,900 | 614,154 | 1.52 | 0.01ms | 19.38ms | 13.81ms | 27.36ms |
| **tiktoken (cl100k/GPT-4)** | 100,000 | 9,000 | 578,216 | 1.89 | 0.01ms | 15.57ms | 12.40ms | 20.60ms |
| **HF LLaMA (SP-BPE)** | 32,000 | 11,401 | 389,845 | 1.00 | 1357.85ms | 29.24ms | 25.61ms | 36.08ms |
| **HF T5 (SentencePiece)** | 32,000 | 12,601 | 314,385 | 0.73 | 1820.48ms | 40.08ms | 34.52ms | 49.43ms |
| **HF BERT (WordPiece)** | 30,522 | 11,402 | 285,649 | 0.74 | 1613.55ms | 39.92ms | 35.08ms | 55.06ms |
| **HF GPT-2 (BPE)** | 50,257 | 15,700 | 263,369 | 0.49 | 1944.51ms | 59.61ms | 44.19ms | 72.57ms |

---

## Visualization

![Benchmark Comparison](benchmark_comparison.png)

---

## Speed Comparison

| Tokenizer | Speed vs CRAYON |
| :--- | ---: |
| **CRAYON (lite, 50k)** | **baseline** |
| tiktoken (p50k/GPT-3) | 33.0x slower |
| tiktoken (cl100k/GPT-4) | 35.1x slower |
| HF LLaMA (SP-BPE) | 52.1x slower |
| HF T5 (SentencePiece) | 64.5x slower |
| HF BERT (WordPiece) | 71.0x slower |
| HF GPT-2 (BPE) | 77.1x slower |

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
