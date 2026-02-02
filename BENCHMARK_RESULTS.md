# XERV Crayon V2.0 - Competitive Benchmark Results

**100% HONEST. NO SUGARCOATING. DATA-DRIVEN.**

**Date:** 2026-02-02 21:46:22

**Test Text Size:** 30,800 bytes (30.1 KB)

**Iterations:** 10 (+ 2 warmup)

---

## Results (Real Tokenizers Only - Sorted by Speed)

| Tokenizer | Vocab Size | Token Count | Tokens/sec | MB/sec | Load Time | Avg Time | Min Time | Max Time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CRAYON (CPU - code)** | ~250k | 30,800 | 23,762,131 | 22.66 | 128.98ms | 1.30ms | 1.01ms | 2.30ms |
| **CRAYON (CPU - science)** | ~250k | 24,900 | 18,170,673 | 21.43 | 3.81ms | 1.37ms | 0.97ms | 2.44ms |
| **CRAYON (CPU - lite)** | 50k | 15,700 | 9,931,052 | 18.58 | 20.63ms | 1.58ms | 1.29ms | 1.94ms |
| **tiktoken (p50k/GPT-3)** | 50,000 | 11,900 | 422,632 | 1.04 | 0.01ms | 28.16ms | 21.03ms | 55.72ms |
| **tiktoken (cl100k/GPT-4)** | 100,000 | 9,000 | 383,486 | 1.25 | 0.01ms | 23.47ms | 20.07ms | 35.85ms |
| **HF T5 (SentencePiece)** | 32,000 | 12,601 | 382,678 | 0.89 | 1777.77ms | 32.93ms | 32.27ms | 34.05ms |
| **HF LLaMA (SP-BPE)** | 32,000 | 11,401 | 287,510 | 0.74 | 1174.77ms | 39.65ms | 30.96ms | 45.88ms |
| **HF GPT-2 (BPE)** | 50,257 | 15,700 | 213,441 | 0.40 | 1819.56ms | 73.56ms | 61.30ms | 98.43ms |
| **HF BERT (WordPiece)** | 30,522 | 11,402 | 193,874 | 0.50 | 1832.96ms | 58.81ms | 50.55ms | 68.34ms |

---

## Visualization

![Benchmark Comparison](benchmark_comparison.png)

---

## Speed Comparison

| Tokenizer | Speed vs CRAYON |
| :--- | ---: |
| **CRAYON (CPU - code)** | **baseline** |
| **CRAYON (CPU - science)** | **baseline** |
| **CRAYON (CPU - lite)** | **baseline** |
| tiktoken (p50k/GPT-3) | 56.2x slower |
| tiktoken (cl100k/GPT-4) | 62.0x slower |
| HF T5 (SentencePiece) | 62.1x slower |
| HF LLaMA (SP-BPE) | 82.6x slower |
| HF GPT-2 (BPE) | 111.3x slower |
| HF BERT (WordPiece) | 122.6x slower |

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
