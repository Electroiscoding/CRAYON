# XERV Crayon V2.0 - Competitive Benchmark Results

**100% HONEST. NO SUGARCOATING. DATA-DRIVEN.**

**Date:** 2026-01-25 23:32:20

**Test Text Size:** 30,800 bytes (30.1 KB)

**Iterations:** 10 (+ 2 warmup)

---

## Results (Real Tokenizers Only - Sorted by Speed)

| Tokenizer | Vocab Size | Token Count | Tokens/sec | MB/sec | Load Time | Avg Time | Min Time | Max Time |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **CRAYON (CPU - science)** | ~250k | 24,900 | 21,102,590 | 24.89 | 0.77ms | 1.18ms | 1.03ms | 1.41ms |
| **CRAYON (CPU - code)** | ~250k | 22,100 | 14,255,305 | 18.95 | 0.56ms | 1.55ms | 1.38ms | 1.78ms |
| **CRAYON (CPU - lite)** | 50k | 15,700 | 10,251,187 | 19.18 | 0.96ms | 1.53ms | 1.08ms | 1.92ms |
| **tiktoken (p50k/GPT-3)** | 50,000 | 11,900 | 356,664 | 0.88 | 0.01ms | 33.36ms | 27.52ms | 50.98ms |
| **tiktoken (cl100k/GPT-4)** | 100,000 | 9,000 | 315,068 | 1.03 | 0.01ms | 28.57ms | 22.97ms | 49.09ms |
| **HF GPT-2 (BPE)** | 50,257 | 15,700 | 289,974 | 0.54 | 1755.15ms | 54.14ms | 45.87ms | 60.18ms |
| **HF LLaMA (SP-BPE)** | 32,000 | 11,401 | 210,363 | 0.54 | 1712.58ms | 54.20ms | 44.13ms | 75.19ms |
| **HF T5 (SentencePiece)** | 32,000 | 12,601 | 184,227 | 0.43 | 1844.30ms | 68.40ms | 53.73ms | 93.09ms |
| **HF BERT (WordPiece)** | 30,522 | 11,402 | 166,747 | 0.43 | 1531.15ms | 68.38ms | 41.35ms | 109.05ms |

---

## Visualization

![Benchmark Comparison](benchmark_comparison.png)

---

## Speed Comparison

| Tokenizer | Speed vs CRAYON |
| :--- | ---: |
| **CRAYON (CPU - science)** | **baseline** |
| **CRAYON (CPU - code)** | **baseline** |
| **CRAYON (CPU - lite)** | **baseline** |
| tiktoken (p50k/GPT-3) | 59.2x slower |
| tiktoken (cl100k/GPT-4) | 67.0x slower |
| HF GPT-2 (BPE) | 72.8x slower |
| HF LLaMA (SP-BPE) | 100.3x slower |
| HF T5 (SentencePiece) | 114.5x slower |
| HF BERT (WordPiece) | 126.6x slower |

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
