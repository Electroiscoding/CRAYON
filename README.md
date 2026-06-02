# Crayon 🖍️
Crayon is a high-performance, hardware-accelerated tokenizer engineered for instant vocabulary swapping and maximum throughput.
Designed to eliminate the bottleneck of data preprocessing in LLM pipelines, Crayon operates using a unique **cartridge system**—pre-built vocabulary profiles that can be loaded and swapped instantly. This allows developers to seamlessly switch between 50k and 250k vocabularies without rebuilding the tokenizer state.

# Introduction
CRAYON is a tokenizer that uses a cartridge system.A cartridge is a pre-built vocabulary profile that can be swapped instantly.This allows switching between 50k and 200k vocab easy. Written in C++17 with linked-list BPE (Byte Pair Encoding) for training.Native GPU kernels in CUDA (NVIDIA) and HIP (AMD).Supports CPU with AVX2 & AVX-512 SIMD.Uses zero-copy mmap loading of .DAT files for instant startup.

# Architecture

This part provides a highly detailed systems-level architectural breakdown of the **XERV Crayon** tokenizer. Crayon is designed to solve the data-loading and runtime bottlenecks of subword tokenization in LLM pipelines by using hardware-accelerated Double-Array Tries (DAT), zero-copy memory mapping, and dynamic cartridge profiles.

---

## 1. Executive Summary & System Overview

Crayon transitions subword tokenization from traditional pointer-heavy trees or lookup tables to a rigid, memory-aligned data layout. By compiling a vocabulary into a **Double-Array Trie (DAT)** format, Crayon reduces transitions to $O(1)$ flat array accesses. 

```mermaid
graph TD
    classDef layer fill:#f9f,stroke:#333,stroke-width:2px;
    classDef env fill:#e1f5fe,stroke:#01579b,stroke-dasharray: 5 5;

    Resource[Resources Layer: Corpora & JSONs] -->|BPE Train| Trainer[BPE Trainer: C++ Parallel Arrays]
    Trainer -->|Save Merges| JSON[Trained Vocab JSON]
    JSON -->|Compile| Compiler[DAT Compiler: First-Fit Search]
    Compiler -->|Serialize| Cartridge[Cartridges: .dat & .json Profiles]
    
    Cartridge -->|Zero-Copy mmap| Frontend[Python Frontend: CrayonVocab]
    
    subgraph Frontend ["CrayonVocab Unified Interface"]
        Frontend -->|Auto-Detect / Select| Backend{Backend Router}
        Backend -->|AVX2 SIMD / Fallback| CPU[CPU Engine: cpu_engine.cpp]
        Backend -->|NVCC Kernels| CUDA[CUDA Engine: gpu_engine_cuda.cu]
        Backend -->|HIPCC Kernels| ROCm[ROCm Engine: rocm_engine.hip]
    end

    class Resource,Trainer,Compiler,Cartridge,Frontend layer;
```

---

## 2. Active Profiles (The Cartridge System)

At the core of Crayon's versatility is its **Cartridge System** defined in [profiles.py](/core/profiles.py). Instead of hardcoding vocabulary layouts or loading slow string dicts, Crayon represents vocabularies as pre-built binary profiles loaded instantly via zero-copy memory mapping (`mmap`).

### 2.1 Primary Pre-Packaged Profiles
Crayon ships with two canonical production cartridges stored in [src/crayon/resources/dat/](/src/crayon/resources/dat/):

1. **Lite (`lite`)**
   - **File Paths**: [vocab_lite.dat](/src/crayon/resources/dat/vocab_lite.dat) and [vocab_lite.json](/src/crayon/resources/dat/vocab_lite.json)
   - **Target Size**: 50,000 subwords
   - **Disk Sizes**: DAT: ~1.17 MB, JSON: ~520 KB
   - **Use Case**: General-purpose LLM tokenization (equivalent to `tiktoken` 50k layouts) with low memory footprints.

2. **Standard (`standard`)**
   - **File Paths**: [vocab_standard.dat](/src/crayon/resources/dat/vocab_standard.dat) and [vocab_standard.json](/src/crayon/resources/dat/vocab_standard.json)
   - **Target Size**: 250,000 subwords
   - **Disk Sizes**: DAT: ~5.23 MB, JSON: ~2.28 MB
   - **Use Case**: Rich multi-domain/multilingual vocabularies requiring massive context representation.


### 2.2 Profile Resolution Strategy
When loading a profile via `vocab.load_profile("profile_name")`, the engine executes an ordered path resolution defined in `_get_profile_search_paths()` in [vocabulary.py](/src/crayon/core/vocabulary.py):
1. **Package Resources**: `src/crayon/resources/dat/vocab_{profile_name}.dat`
2. **Modern importlib.resources**: Interrogates Python's modern package structure.
3. **`CRAYON_PROFILE_DIR`**: Local developer directory override.
4. **User Home Cache**: `~/.cache/xerv/crayon/profiles/`
5. **System Cache (Linux only)**: `/var/cache/crayon/`

---

## 3. Double-Array Trie (DAT) Data Layout

To eliminate pointer chasing and dynamic hash lookups, Crayon implements a unified, cache-aligned Double-Array Trie representation.

### 3.1 Binary Format (Serialized `.dat` File)
A compiled profile is serialized to disk with the following binary structure:
- **Header (12 bytes)**:
  - `[0-3]`: Magic bytes `0x59415243` ("CRAY" in ASCII/little-endian)
  - `[4-7]`: Version integer (Version 2)
  - `[8-11]`: Total nodes ($N$)
- **Data Section**:
  - `[12 to (12 + N * 4)]`: `BASE` Array ($N \times \text{int32}$)
  - `[(12 + N * 4) to (12 + 2N * 4)]`: `CHECK` Array ($N \times \text{int32}$)
  - `[(12 + 2N * 4) to (12 + 3N * 4)]`: `VALUES` Array ($N \times \text{int32}$)

### 3.2 Transition Mathematics
For a parent state $s$ and an input byte value $c$:
1. The target child slot $t$ is calculated via base offset shift:
   $$t = \text{BASE}[s] + c$$
2. The transition is verified by checking ownership in the parent array:
   $$\text{CHECK}[t] == s$$
3. If valid, the engine traverses to state $t$. If `VALUES[t] != -1`, a vocabulary match of token ID `VALUES[t]` is captured, updating the longest match bounds.

---

## 4. Systems Architecture: Component-by-Component Analysis

### 4.1 C++ DAT Compiler (First-Fit Search)
Located in [compiler.cpp](/src/crayon/c_ext/compiler.cpp), this native C++ class achieves a **~500x speedup** over naive Python-based trie packers.

- **Temporary Trie Structure**: Converts the input list of vocabulary strings into a classic node-and-pointer trie (`TrieNode`).
- **Array Packing (First-Fit Scan)**:
  1. For each node, gathers all children bytes: $\{c_1, c_2, ..., c_k\}$.
  2. Commences a linear search starting from base index $b = 1$ to find the first index where slots $b + c_i$ are empty (`CHECK[b + c_i] == -1` for all $i$).
  3. If a collision occurs, it increments $b$ and retries.
  4. Once a collision-free offset $b$ is located, it sets `BASE[parent] = b` and claims the slots by setting `CHECK[b + c_i] = parent`.
- **Dynamic Resize**: The vectors dynamically scale in chunks of 512 integers to minimize allocation overhead.
- **GIL Release**: Releases Python's GIL during the heavy tree traversal loop.

### 4.2 CPU Inference Engine (SIMD Vectorized)
Located in [cpu_engine.cpp](src/crayon/c_ext/cpu_engine.cpp), the CPU engine acts as the low-latency fallback and single-doc execution loop.

- **AVX2 ASCII Verification**: Reads 32 bytes of text in a single CPU cycle using AVX2 intrinsics:
  ```cpp
  inline int is_ascii_32_avx2(const char* ptr) {
      __m256i chunk = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(ptr));
      int mask = _mm256_movemask_epi8(chunk);
      return mask == 0;
  }
  ```
- **Dual execution paths**:
  1. **Fast-path**: If the next 32 bytes are pure ASCII, the engine bypasses UTF-8 boundary checks, allowing aggressive compiler loop unrolling.
  2. **Safe UTF-8 path**: Performs character validation at standard boundaries.
- **Zero-Copy Interoperability**: Maps files directly to the CPU using Python's `Py_buffer` protocol, avoiding memory copying into the heap.

### 4.3 GPU Inference Engine (CUDA & ROCm)
- **CUDA Backend**: Located in [gpu_engine_cuda.cu](src/crayon/c_ext/gpu_engine_cuda.cu). Maps one document/sentence to a single CUDA thread. The entire DAT (`base`, `check`, `values`) is loaded into global device VRAM. Threads perform lookahead up to 128 characters to process subword sequences without block synchronizations. Memory allocations use synchronous `cudaMalloc` for PyTorch co-existence.
- **ROCm/HIP Backend**: Located in [rocm_engine.hip](src/crayon/c_ext/rocm_engine.hip). Rewritten to support AMD ROCm architectures natively, utilizing the `hipcc` compiler. Includes proper AMD context cleanup and error diagnostics using `hipGetErrorString()`.

---

## 5. Algorithmic Deep Dive: Hyper-Fast BPE Trainer

Located in [trainer.cpp](src/crayon/c_ext/trainer.cpp), the training engine implements a mathematically optimal, exact greedy BPE training algorithm on a single CPU core. It solves the $O(N)$ text scanning bottleneck by utilizing a sophisticated tri-structure model:

### 5.1 Parallel Array Linked-List
The training corpus is represented inside the CPU cache as four contiguous arrays of size $N$ (corpus length):
1. `tokens` (int32): Stores the token ID of the subword at position $i$.
2. `prev_pos` (int32): Pointer index to the previous active token.
3. `next_pos` (int32): Pointer index to the next active token.
4. `active` (bool): Boolean flag tracking whether position $i$ is alive or has been merged.

By avoiding pointer structs, Crayon fits large arrays into the L3 cache. Merging adjacent tokens becomes a constant-time pointer rearrangement:
```cpp
next_pos[pos] = next_next_idx;
if (next_next_idx != -1) {
    prev_pos[next_next_idx] = pos;
}
active[next_idx] = false;
```

### 5.2 Inverted Index (`pair_locations`)
A hash map combines byte pairs `(TokenA, TokenB)` using Knuth's multiplicative hash (`first * 31 + second`) and maps them to a dynamic vector of indices where they occur. This eliminates corpus-wide rescanning; the trainer jumps directly to the merge sites.

### 5.3 Lazy Max-Heap
A priority queue stores `{count, pair}` tuples. When adjacent merges disrupt sibling pairs (decreasing their true occurrence count), Crayon leaves the stale count in the queue. Upon popping, the trainer validates the popped count against the true count in the hash map. If they mismatch, it is discarded as a "lazy skip" in $O(1)$ time, reducing heap rearrangements from $O(N)$ to $O(\log H)$ amortized.

---

## 6. Memory Models & Concurrency

To minimize garbage collection pressure and multithreading overhead, Crayon implements several advanced concurrent memory wrappers:

- **Zero-Copy Memory Mapping**: Located in [zerocopy.py](src/crayon/memory/zerocopy.py). Leverages Python's `mmap` module combined with `memoryview` slicing. Large files are tokenized by mapping chunks directly to the OS page cache without loading the whole file into RAM.
- **Memory Pool**: Located in [pool.py](src/crayon/memory/pool.py). A thread-safe pool of reusable `bytearray` chunks (64KB default matching the L2 cache size) which mitigates heap allocation churn.
- **Lock-Free Cache**: Located in [cache.py](src/crayon/memory/cache.py). A thread-safe L1 cache using optimistic concurrency. Reads verify a `version` counter before and after reading keys/values to detect race conditions (preventing the ABA problem) without locks.
- **Thread-Local State**: Located in [thread_local.py](src/crayon/concurrency/thread_local.py). Dedicates a private `LockFreeVocabCache` (capacity 2048) and workspace buffer to each thread, avoiding thread contention and cache false-sharing.
- **Multi-stage Pipeline Tokenizer**: Located in [pipeline.py](src/crayon/concurrency/pipeline.py). Establishes a 3-stage multithreaded queue:
  1. `Stage-Normalize`: Runs fast ASCII-optimistic Unicode NFC normalization.
  2. `Stage-Tokenize`: Invokes the native C++ Crayon backend.
  3. `Stage-Format`: Bundles token IDs into model-compliant dictionary inputs.

---

## 7. Performance Benchmarks

Captured on standard commodity x86_64 hardware with a 68.4 KB mixed corpus:

### 7.1 Throughput Comparison (Tokens / Second)
| Implementation | English | Code | Unicode | Mixed Text |
| :--- | ---: | ---: | ---: | ---: |
| **Crayon (`lite`, 50k)** | **18,407,951** | **33,161,787** | **43,921,330** | **24,589,766** |
| **Crayon (`standard`, 250k)**| **17,154,914** | **18,707,550** | **29,227,498** | **17,394,245** |
| tiktoken (`cl100k_base`) | 1,198,631 | 916,869 | 1,696,065 | 1,066,657 |
| HF GPT-2 Tokenizer | 237,117 | — | — | — |

*Crayon is roughly **10x to 35x** faster than Rust-based tiktoken on CPU, especially on code syntax and Unicode documents which benefit from the AVX2 fast-path and cache-aligned memory access.*

### 7.2 Latency Characteristics
- **Profile Initialization (Cold Load)**: **0.54 ms** (vs. ~1,200 ms to 2,100 ms for standard JSON parsing tokenizers) via instant `mmap`.
- **Cartridge Compilation (Science Profile, 367 tokens)**: **38 ms** via native First-Fit C++ assembler.

### 7.3 Extreme Stress Test (~100M Characters)

To evaluate the absolute throughput limit and memory stability of Crayon under heavy load, we executed a benchmark tokenizing a massive text block of **~100 million characters** (~95.37 MiB) using the `standard` (200k) profile:

```
===========================================================================
               CRAYON TOKENIZER EXTREME STRESS TEST RESULTS                
===========================================================================
Device       | Total Tokens       | Time (s)        | Throughput        | Performance
---------------------------------------------------------------------------
CPU          | 21,212,115         | 0.8899          | 107.16 MiB/s      | 23.84 M tokens/s
CUDA         | 21,212,115         | 17.3300         | 5.50 MiB/s        | 1.22 M tokens/s
===========================================================================
```

*Note on the CUDA/ROCm Truncation Bug:*
- **Root Cause**: The GPU backends ([gpu_engine_cuda.cu](src/crayon/c_ext/gpu_engine_cuda.cu) and [rocm_engine.hip](src/crayon/c_ext/rocm_engine.hip)) calculated the output buffer token capacity per sequence based on the batch average sentence length, but enforced a strict hardcoded limit of `4096` tokens. Any sequence generating more than 4,096 tokens was silently truncated, leading to data loss on large inputs.
- **What Changed (Fix)**: The static `4096` cap was replaced with dynamic capacity planning. The engine now determines the maximum sentence length in the batch (`max_len`) and sets sequence capacity to `max_len + 64`. To prevent GPU Out-Of-Memory (OOM) failures under massive batch workloads, it enforces a dynamic safety ceiling based on a total allocation budget of 512M elements (~2 GB). This enables processing massive individual documents safely.

