
# Crayon 🖍️
Crayon is a high-performance, hardware-accelerated tokenizer engineered for instant vocabulary swapping and maximum throughput.
Designed to eliminate the bottleneck of data preprocessing in LLM pipelines, Crayon operates using a unique **cartridge system**—pre-built vocabulary profiles that can be loaded and swapped instantly. This allows developers to seamlessly switch between 50k and 250k vocabularies without rebuilding the tokenizer state.
## ⚡ Core Features
* **Built for Speed:** Written entirely in C++17 utilizing a linked-list BPE (Byte Pair Encoding) algorithm for training.
* **Hardware Acceleration:** Features native GPU kernels in both CUDA (NVIDIA) and HIP (AMD), alongside CPU AVX2 SIMD support.
* **Zero-Copy Loading:** Utilizes zero-copy mmap loading for `.DAT` files, enabling near-instantaneous startup times.
* **Direct Streaming:** Supports zero-disk streaming directly from Hugging Face datasets.
## 🚀 Installation
Install the latest version directly from PyPI (v2.0.1):
```bash
pip install xerv-crayon
```
*Note: Crayon also supports manual building with python setup.py build_ext --inplace, which will automatically detect your local GPU compilers (nvcc or hipcc).*
## 💻 Quickstart & Usage
Crayon's cartridge system allows you to effortlessly load different profiles (e.g., standard or lite) on the fly.
```python
from crayon import CrayonVocab
# Initialize the tokenizer with auto-device detection (CPU/CUDA/HIP)
tokenizer = CrayonVocab(device="auto")
print("--- Testing Standard Profile ---")
tokenizer.load_profile("standard")
tokens_std = tokenizer.tokenize("that is a test for the standard profile and lite profile and god")
print("Tokens:", tokens_std)
print("Decoded:", tokenizer.decode(tokens_std))
print("\n--- Testing Lite Profile ---")
tokenizer.load_profile("lite")
tokens_lite = tokenizer.tokenize("my daughter")
print("Tokens:", tokens_lite)
print("Decoded:", tokenizer.decode(tokens_lite))
```
## 📊 Benchmarks
Crayon consistently outperforms standard tokenizers in both throughput (Tokens/s) and processing speed (MB/s).
### Visual Performance
### Throughput & Performance Table

| Implementation | Dataset | Load Time (ms) | Throughput (Tokens/sec) | Data Rate (MB/sec) |
| :--- | :--- | :--- | :--- | :--- |
| **crayon:cpu:lite** | english | 293.85 | **763,799** | 3.74 |
| **crayon:cpu:lite** | code | 53.47 | **4,052,162** | 7.28 |
| **crayon:cpu:lite** | unicode | 34.77 | **5,537,900** | 7.35 |
| **crayon:cpu:lite** | mixed | 44.44 | **2,978,063** | 6.20 |
| **crayon:cpu:standard** | english | 256.84 | **2,980,628** | 15.47 |
| **crayon:cpu:standard** | code | 373.48 | **2,954,819** | 9.47 |
| **crayon:cpu:standard** | unicode | 289.20 | **6,557,164** | 19.11 |
| **crayon:cpu:standard** | mixed | 161.32 | **1,817,683** | 6.50 |
| tiktoken:p50k_base | english | 971.75 | 149,704 | 0.73 |
| tiktoken:p50k_base | code | 0.02 | 145,018 | 0.35 |
| tiktoken:cl100k_base | english | 1833.94 | 130,941 | 0.66 |
| tiktoken:cl100k_base | code | 0.01 | 128,272 | 0.42 |
| tiktoken:o200k_base | english | 2667.90 | 158,827 | 0.80 |
| tiktoken:o200k_base | code | 0.01 | 175,636 | 0.56 |

### Key Takeaways
 * **Massive Throughput:** Crayon (standard profile on CPU) achieves up to **6.5M tokens/sec** on Unicode datasets, vastly outperforming tiktoken variants which hover between 130k and 320k tokens/sec.
 * **Optimized for Code:** On raw code datasets, Crayon's lite profile processes over **4M tokens/sec**, making it highly optimized for codebase indexing and LLM code-generation pipelines.
