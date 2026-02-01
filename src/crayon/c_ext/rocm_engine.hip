/*
 * XERV CRAYON ROCm ENGINE (AMD BACKEND) v4.3.0
 * ============================================
 * Architecture: CDNA/RDNA Optimized HIP Kernel
 * Target Hardware: AMD Instinct MI250/MI300, Radeon RX 7000+
 * 
 * ENGINEERING DEEP DIVE:
 * 1. Coalesced Memory Access: Threads align reads to 128-byte cache lines.
 * 2. Wavefront Synchronization: Minimized control flow divergence.
 * 3. Zero-Copy IO: Uses pinned host memory where applicable for transfer.
 * 
 * COMPILATION NOTES:
 * This file MUST be compiled with hipcc (AMD's HIP compiler).
 * File extension .hip ensures proper compiler invocation.
 */

#include <hip/hip_runtime.h>
#include <Python.h>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>

// --- MACRO FOR SAFE HIP CALLS ---
#define HIP_SAFE_CALL(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        const char* errStr = hipGetErrorString(err); \
        PyErr_Format(PyExc_RuntimeError, "HIP Error: %s at %s:%d", errStr, __FILE__, __LINE__); \
        return NULL; \
    } \
} while(0)

#define HIP_SAFE_CALL_VOID(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

// --- HOST FUNCTION: GET HARDWARE INFO ---
static PyObject* get_hardware_info(PyObject* self, PyObject* args) {
    int deviceId = 0;
    hipError_t err = hipGetDevice(&deviceId);
    if (err != hipSuccess) {
        return PyUnicode_FromString("AMD ROCm (Device Not Found)");
    }

    hipDeviceProp_t prop;
    err = hipGetDeviceProperties(&prop, deviceId);
    if (err != hipSuccess) {
        return PyUnicode_FromString("AMD ROCm (Properties Unavailable)");
    }

    // Format: "AMD Radeon RX 7900 XTX [Arch 11.0, 24576 MB VRAM]"
    std::string info = std::string(prop.name) + " [Arch " + 
                       std::to_string(prop.major) + "." + std::to_string(prop.minor) + ", " +
                       std::to_string(prop.totalGlobalMem / (1024*1024)) + " MB VRAM]";
                       
    return PyUnicode_FromString(info.c_str());
}

// --- PERSISTENT HBM STORAGE (Device Globals) ---
// These pointers reference data living in the AMD GPU's High Bandwidth Memory.
// They are static to maintain state between Python function calls.
static int32_t *d_rocm_base = nullptr;
static int32_t *d_rocm_check = nullptr;
static int32_t *d_rocm_values = nullptr;
static uint32_t rocm_trie_size = 0;
static bool rocm_loaded = false;
static bool rocm_initialized = false;

// --- CLEANUP ---
static void cleanup_rocm_memory(void) {
    if (d_rocm_base) { hipFree(d_rocm_base); d_rocm_base = nullptr; }
    if (d_rocm_check) { hipFree(d_rocm_check); d_rocm_check = nullptr; }
    if (d_rocm_values) { hipFree(d_rocm_values); d_rocm_values = nullptr; }
    rocm_loaded = false;
    rocm_trie_size = 0;
}

// --- THE HIP KERNEL (The "Workhorse") ---
// Runs on the GPU Compute Units (CU).
// __global__ indicates this function is callable from the Host (CPU) but executes on the Device (GPU).
__global__ void tokenize_kernel_hip(
    const int32_t* __restrict__ base,    // Cached in L1 Texture Cache
    const int32_t* __restrict__ check,   // Cached in L1 Texture Cache
    const int32_t* __restrict__ values,  // Cached in L1 Texture Cache
    const char* __restrict__ text_pool,  // Massive contiguous char buffer
    const int* __restrict__ offsets,     // Start/End indices for each string
    int* out_tokens,                     // Flattened Output Buffer
    int* out_counts,                     // Token count per sentence
    int n_sentences,
    int max_capacity,                    // Hard limit on tokens per sequence (e.g., 2048)
    uint32_t trie_sz                     // Trie size for bounds checking
) {
    // 1. Calculate Global Thread Identity
    // HIP uses the same coordinate system as CUDA: GlobalID = BlockID * BlockDim + ThreadID
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check: Ensure we don't read past the number of sentences
    if (idx >= n_sentences) return;

    // 2. Fetch Sentence Boundaries
    // Reading 'offsets' is coalesced; adjacent threads read adjacent integers.
    int start = offsets[idx];
    int end = offsets[idx+1];
    int len = end - start;
    
    // 3. Initialize Local Register State
    // We keep 'node', 'count', and 'pos' in VGPRs (Vector General Purpose Registers)
    // to avoid latency penalties from accessing global memory.
    int count = 0;
    int write_ptr = idx * max_capacity; // Pre-calculated offset for this thread's output

    int pos = 0;
    
    // 4. Tokenization Loop (The Critical Path)
    // We iterate until the end of the string or until we hit the context limit.
    while (pos < len && count < max_capacity) {
        int best_token = 1; // Default to UNK (ID 1)
        int best_len = 0;
        int curr = 0;       // Start from root
        
        // Inner Loop: Traverses the Trie structure for the longest match
        // WARNING: This is where Wavefront Divergence occurs. Threads processing short words
        // will wait for threads processing long words. We mitigate this by keeping the loop body tight.
        for (int i = pos; i < len && i < pos + 128; ++i) {  // Max 128 chars lookahead
            unsigned char c = (unsigned char)text_pool[start + i];
            
            // Branchless Base Lookup
            // The 'base' array is heavily accessed, so it stays hot in the L2 cache.
            int next = base[curr] + c;
            
            // Check Transition Validity with bounds checking
            if (next >= 0 && (uint32_t)next < trie_sz && check[next] == curr) {
                curr = next;
                
                // Check if this node marks a valid token
                int val = values[curr];
                // values[curr] == -1 means intermediate node (not a token end)
                if (val != -1) {
                    best_token = val;
                    best_len = (i - pos) + 1;
                }
            } else {
                break;
            }
        }
        
        // 5. Commit Result
        out_tokens[write_ptr + count] = best_token;
        count++;
        pos += (best_len > 0) ? best_len : 1;
    }
    
    // Write final token count for this sentence
    out_counts[idx] = count;
}

// --- INIT ROCM DEVICE ---
static PyObject* init_rocm_device(void) {
    if (rocm_initialized) {
        Py_RETURN_TRUE;
    }
    
    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        PyErr_SetString(PyExc_RuntimeError, "No ROCm/HIP devices available");
        return NULL;
    }
    
    // Set device 0 and force context creation
    err = hipSetDevice(0);
    if (err != hipSuccess) {
        PyErr_Format(PyExc_RuntimeError, "Failed to set HIP device: %s", hipGetErrorString(err));
        return NULL;
    }
    
    // Force context initialization with a dummy allocation
    void* dummy = nullptr;
    err = hipMalloc(&dummy, 1);
    if (err != hipSuccess) {
        PyErr_Format(PyExc_RuntimeError, "Failed to initialize HIP context: %s", hipGetErrorString(err));
        return NULL;
    }
    hipFree(dummy);
    
    rocm_initialized = true;
    Py_RETURN_TRUE;
}

// --- HOST FUNCTION: LOAD DICTIONARY (One-Time) ---
// Transfers the Double-Array Trie from System RAM to GPU VRAM/HBM.
static PyObject* load_rocm(PyObject* self, PyObject* args) {
    PyObject* py_bytes;
    if (!PyArg_ParseTuple(args, "O", &py_bytes)) return NULL;
    
    if (!PyBytes_Check(py_bytes)) {
        PyErr_SetString(PyExc_TypeError, "Expected bytes object");
        return NULL;
    }

    // Step 1: Initialize ROCm if not done
    if (!rocm_initialized) {
        PyObject* init_result = init_rocm_device();
        if (init_result == NULL) {
            return NULL;  // Error already set
        }
        Py_DECREF(init_result);
    }

    // Step 2: Parse DAT file header
    Py_ssize_t total_len = PyBytes_Size(py_bytes);
    if (total_len < 12) {
        PyErr_SetString(PyExc_ValueError, "DAT file too small (< 12 bytes)");
        return NULL;
    }

    const char* raw = PyBytes_AsString(py_bytes);
    
    // Read trie size from offset 8 (standard DAT format)
    uint32_t sz = 0;
    memcpy(&sz, raw + 8, sizeof(uint32_t));
    
    // Validate size
    if (sz == 0) {
        PyErr_SetString(PyExc_ValueError, "Trie size is 0");
        return NULL;
    }
    if (sz > (1u << 24)) {  // Max 16M entries
        PyErr_SetString(PyExc_ValueError, "Trie size exceeds maximum (16M entries)");
        return NULL;
    }

    size_t array_bytes = sz * sizeof(int32_t);
    size_t required_bytes = 12 + (array_bytes * 3);
    
    if ((size_t)total_len < required_bytes) {
        PyErr_Format(PyExc_ValueError, 
                     "DAT file incomplete. Need %zu bytes, got %zd", 
                     required_bytes, total_len);
        return NULL;
    }

    // Step 3: Cleanup any previous allocations
    cleanup_rocm_memory();

    // Step 4: Allocate HBM (High Bandwidth Memory)
    hipError_t err;
    
    err = hipMalloc((void**)&d_rocm_base, array_bytes);
    if (err != hipSuccess) {
        cleanup_rocm_memory();
        PyErr_Format(PyExc_RuntimeError, "hipMalloc d_rocm_base failed: %s", hipGetErrorString(err));
        return NULL;
    }
    
    err = hipMalloc((void**)&d_rocm_check, array_bytes);
    if (err != hipSuccess) {
        cleanup_rocm_memory();
        PyErr_Format(PyExc_RuntimeError, "hipMalloc d_rocm_check failed: %s", hipGetErrorString(err));
        return NULL;
    }

    err = hipMalloc((void**)&d_rocm_values, array_bytes);
    if (err != hipSuccess) {
        cleanup_rocm_memory();
        PyErr_Format(PyExc_RuntimeError, "hipMalloc d_rocm_values failed: %s", hipGetErrorString(err));
        return NULL;
    }

    // Step 5: Transfer Host -> Device
    const char* data_ptr = raw + 12;
    
    err = hipMemcpy(d_rocm_base, data_ptr, array_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        cleanup_rocm_memory();
        PyErr_Format(PyExc_RuntimeError, "hipMemcpy d_rocm_base failed: %s", hipGetErrorString(err));
        return NULL;
    }
    
    err = hipMemcpy(d_rocm_check, data_ptr + array_bytes, array_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        cleanup_rocm_memory();
        PyErr_Format(PyExc_RuntimeError, "hipMemcpy d_rocm_check failed: %s", hipGetErrorString(err));
        return NULL;
    }
    
    err = hipMemcpy(d_rocm_values, data_ptr + (array_bytes * 2), array_bytes, hipMemcpyHostToDevice);
    if (err != hipSuccess) {
        cleanup_rocm_memory();
        PyErr_Format(PyExc_RuntimeError, "hipMemcpy d_rocm_values failed: %s", hipGetErrorString(err));
        return NULL;
    }
    
    // Step 6: Sync and verify
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        cleanup_rocm_memory();
        PyErr_Format(PyExc_RuntimeError, "hipDeviceSynchronize failed: %s", hipGetErrorString(err));
        return NULL;
    }
    
    rocm_trie_size = sz;
    rocm_loaded = true;
    
    // Return success info
    char msg[256];
    snprintf(msg, sizeof(msg), "Loaded %u entries (%.2f MB) to AMD GPU", 
             sz, (array_bytes * 3) / (1024.0 * 1024.0));
    return PyUnicode_FromString(msg);
}

// --- HOST FUNCTION: BATCH EXECUTE ---
// Prepares input data and launches the HIP kernel.
static PyObject* tokenize_batch_rocm(PyObject* self, PyObject* args) {
    PyObject* list_obj;
    if (!PyArg_ParseTuple(args, "O", &list_obj)) return NULL;
    
    if (!PyList_Check(list_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected list of strings");
        return NULL;
    }
    
    Py_ssize_t n = PyList_Size(list_obj);
    if (n == 0) return PyList_New(0);

    // Check engine state
    if (!rocm_loaded || !d_rocm_base || !d_rocm_check || !d_rocm_values) {
        PyErr_SetString(PyExc_RuntimeError, "ROCm engine not loaded. Call load_rocm() first.");
        return NULL;
    }

    // 1. Flatten Strings (CPU Pre-processing)
    // GPUs cannot handle 'lists of objects'. We must serialize the Python List[str] 
    // into a single contiguous char buffer (pool) and an offset array.
    std::vector<char> pool;
    std::vector<int> offsets;
    offsets.reserve(n + 1);
    
    size_t total_chars = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* s = PyList_GetItem(list_obj, i);
        if (!PyUnicode_Check(s)) {
            PyErr_SetString(PyExc_TypeError, "List must contain only strings");
            return NULL;
        }
        
        Py_ssize_t len;
        const char* p = PyUnicode_AsUTF8AndSize(s, &len);
        if (!p) return NULL;
        
        offsets.push_back((int)total_chars);
        pool.insert(pool.end(), p, p + len);
        total_chars += len;
    }
    offsets.push_back((int)total_chars);

    // 2. Calculate max tokens per sentence
    size_t avg_len = total_chars / n;
    int max_tok = (int)(avg_len * 2 + 64);
    if (max_tok > 4096) max_tok = 4096;
    if (max_tok < 64) max_tok = 64;

    // 3. Allocate GPU Scratchpads
    char *d_text = nullptr; 
    int *d_offsets = nullptr, *d_out = nullptr, *d_counts = nullptr;
    hipError_t err;
    
    err = hipMalloc((void**)&d_text, pool.size());
    if (err != hipSuccess) {
        PyErr_Format(PyExc_RuntimeError, "hipMalloc d_text failed: %s", hipGetErrorString(err));
        return NULL;
    }
    
    err = hipMalloc((void**)&d_offsets, offsets.size() * sizeof(int));
    if (err != hipSuccess) {
        hipFree(d_text);
        PyErr_Format(PyExc_RuntimeError, "hipMalloc d_offsets failed: %s", hipGetErrorString(err));
        return NULL;
    }
    
    err = hipMalloc((void**)&d_out, n * max_tok * sizeof(int));
    if (err != hipSuccess) {
        hipFree(d_text); hipFree(d_offsets);
        PyErr_Format(PyExc_RuntimeError, "hipMalloc d_out failed: %s", hipGetErrorString(err));
        return NULL;
    }
    
    err = hipMalloc((void**)&d_counts, n * sizeof(int));
    if (err != hipSuccess) {
        hipFree(d_text); hipFree(d_offsets); hipFree(d_out);
        PyErr_Format(PyExc_RuntimeError, "hipMalloc d_counts failed: %s", hipGetErrorString(err));
        return NULL;
    }

    // Zero output buffers
    hipMemset(d_out, 0, n * max_tok * sizeof(int));
    hipMemset(d_counts, 0, n * sizeof(int));

    // 4. Transfer input data
    hipMemcpy(d_text, pool.data(), pool.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), hipMemcpyHostToDevice);

    // 5. Launch Kernel
    // Block Size: 256 is optimal for AMD RDNA/CDNA architectures (4 wavefronts per block).
    // Grid Size: Enough blocks to cover all sentences.
    int threads = 256;
    int blocks = ((int)n + threads - 1) / threads;
    
    // HIP kernel launch syntax
    hipLaunchKernelGGL(tokenize_kernel_hip, dim3(blocks), dim3(threads), 0, 0, 
        d_rocm_base, d_rocm_check, d_rocm_values, 
        d_text, d_offsets, d_out, d_counts, (int)n, max_tok, rocm_trie_size
    );

    // Check for kernel errors
    err = hipGetLastError();
    if (err != hipSuccess) {
        hipFree(d_text); hipFree(d_offsets); hipFree(d_out); hipFree(d_counts);
        PyErr_Format(PyExc_RuntimeError, "Kernel launch failed: %s", hipGetErrorString(err));
        return NULL;
    }

    // 6. Synchronize
    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        hipFree(d_text); hipFree(d_offsets); hipFree(d_out); hipFree(d_counts);
        PyErr_Format(PyExc_RuntimeError, "Kernel execution failed: %s", hipGetErrorString(err));
        return NULL;
    }

    // 7. Retrieve Results
    std::vector<int> h_out(n * max_tok);
    std::vector<int> h_counts(n);
    
    hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(int), hipMemcpyDeviceToHost);
    hipMemcpy(h_counts.data(), d_counts, n * sizeof(int), hipMemcpyDeviceToHost);

    // 8. Build Python result
    PyObject* result = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        int c = h_counts[i];
        PyObject* sub = PyList_New(c);
        int row_ptr = (int)i * max_tok;
        for (int k = 0; k < c; ++k) {
            PyObject* val = PyLong_FromLong(h_out[row_ptr + k]);
            PyList_SetItem(sub, k, val);
        }
        PyList_SetItem(result, i, sub);
    }
    
    // Cleanup
    hipFree(d_text); hipFree(d_offsets); hipFree(d_out); hipFree(d_counts);
    
    // Return tuple (results, metadata)
    PyObject* meta = PyDict_New();
    PyDict_SetItemString(meta, "sentences", PyLong_FromSsize_t(n));
    PyDict_SetItemString(meta, "max_tokens_per_sentence", PyLong_FromLong(max_tok));
    
    PyObject* full_result = PyTuple_New(2);
    PyTuple_SetItem(full_result, 0, result);
    PyTuple_SetItem(full_result, 1, meta);
    
    return full_result;
}

// --- MODULE CLEANUP ---
static void module_cleanup(void* module) {
    cleanup_rocm_memory();
}

// --- MODULE REGISTRATION ---
static PyMethodDef RocmMethods[] = {
    {"load_rocm", load_rocm, METH_VARARGS, "Load DAT into AMD VRAM"},
    {"tokenize_batch_rocm", tokenize_batch_rocm, METH_VARARGS, "HIP Kernel Execute"},
    {"get_hardware_info", get_hardware_info, METH_VARARGS, "Get AMD GPU Telemetry"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef rocm_module = {
    PyModuleDef_HEAD_INIT, 
    "crayon_rocm", 
    "XERV Crayon AMD HIP Backend v4.3.0 - Production Grade", 
    -1, 
    RocmMethods,
    NULL, NULL, NULL,
    module_cleanup
};

PyMODINIT_FUNC PyInit_crayon_rocm(void) {
    return PyModule_Create(&rocm_module);
}
