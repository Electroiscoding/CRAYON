/*
 * XERV CRAYON CUDA ENGINE v3.0 - PRODUCTION GRADE
 * Architecture: Synchronous CUDA with explicit device initialization
 * Target Hardware: NVIDIA Tesla T4/V100/A100/H100
 * Stability: Maximum compatibility - no async allocators, explicit init
 */

#include <cuda_runtime.h>
#include <Python.h>
#include <vector>
#include <cstring>
#include <cstdint>

// --- DEVICE STATE ---
static int32_t *d_base = nullptr;
static int32_t *d_check = nullptr;
static int32_t *d_values = nullptr;
static uint32_t trie_size = 0;
static bool engine_loaded = false;
static bool cuda_initialized = false;

// Forward declarations
static void cleanup_cuda_memory(void);

// --- SAFE CUDA CALL MACRO ---
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        const char* errStr = cudaGetErrorString(err); \
        PyErr_Format(PyExc_RuntimeError, "CUDA Error: %s at %s:%d", errStr, __FILE__, __LINE__); \
        return NULL; \
    } \
} while(0)

// --- SIMPLE TOKENIZATION KERNEL ---
// Uses per-thread local memory instead of shared memory for maximum stability
__global__ void tokenize_kernel(
    const int32_t* __restrict__ base,
    const int32_t* __restrict__ check,
    const int32_t* __restrict__ values,
    const char* __restrict__ text_pool,
    const int* __restrict__ offsets,
    int* out_tokens,
    int* out_counts,
    int n_sentences,
    int max_tokens,
    uint32_t trie_sz
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_sentences) return;

    int start = offsets[idx];
    int end = offsets[idx + 1];
    int len = end - start;
    
    int node = 0;
    int count = 0;
    int write_pos = idx * max_tokens;
    int pos = 0;

    while (pos < len && count < max_tokens) {
        int best_token = 1;  // UNK token
        int best_len = 0;
        int curr = 0;
        
        for (int i = pos; i < len && i < pos + 128; ++i) {  // Max 128 chars lookahead
            unsigned char c = (unsigned char)text_pool[start + i];
            int next = base[curr] + c;
            
            if (next >= 0 && (uint32_t)next < trie_sz && check[next] == curr) {
                curr = next;
                int val = values[curr];
                if (val != -1) {
                    best_token = val;
                    best_len = (i - pos) + 1;
                }
            } else {
                break;
            }
        }
        
        out_tokens[write_pos + count] = best_token;
        count++;
        pos += (best_len > 0) ? best_len : 1;
    }
    
    out_counts[idx] = count;
}

// --- INITIALIZE CUDA DEVICE ---
static PyObject* init_cuda_device(void) {
    if (cuda_initialized) {
        Py_RETURN_TRUE;
    }
    
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        PyErr_SetString(PyExc_RuntimeError, "No CUDA devices available");
        return NULL;
    }
    
    // Set device 0 and force context creation
    err = cudaSetDevice(0);
    if (err != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "Failed to set CUDA device: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    // Force context initialization with a dummy allocation
    void* dummy = nullptr;
    err = cudaMalloc(&dummy, 1);
    if (err != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "Failed to initialize CUDA context: %s", cudaGetErrorString(err));
        return NULL;
    }
    cudaFree(dummy);
    
    cuda_initialized = true;
    Py_RETURN_TRUE;
}

// --- GET HARDWARE INFO ---
static PyObject* get_hardware_info(PyObject* self, PyObject* args) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    
    if (err != cudaSuccess || device_count == 0) {
        return PyUnicode_FromString("No CUDA devices found");
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return PyUnicode_FromString("Failed to get device properties");
    }
    
    char info[512];
    snprintf(info, sizeof(info), "%s [SM %d.%d, %.1f GB VRAM]",
             prop.name, prop.major, prop.minor,
             prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    return PyUnicode_FromString(info);
}

// --- CLEANUP CUDA MEMORY ---
static void cleanup_cuda_memory(void) {
    if (d_base) { cudaFree(d_base); d_base = nullptr; }
    if (d_check) { cudaFree(d_check); d_check = nullptr; }
    if (d_values) { cudaFree(d_values); d_values = nullptr; }
    engine_loaded = false;
    trie_size = 0;
}

// --- LOAD DAT FILE TO GPU ---
static PyObject* load_gpu(PyObject* self, PyObject* args) {
    PyObject* py_bytes;
    if (!PyArg_ParseTuple(args, "O", &py_bytes)) return NULL;
    
    if (!PyBytes_Check(py_bytes)) {
        PyErr_SetString(PyExc_TypeError, "Expected bytes object");
        return NULL;
    }
    
    // Step 1: Initialize CUDA if not done
    if (!cuda_initialized) {
        PyObject* init_result = init_cuda_device();
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
    if (sz > (1 << 24)) {  // Max 16M entries
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
    cleanup_cuda_memory();
    
    // Step 4: Allocate GPU memory (synchronous, most compatible)
    cudaError_t err;
    
    err = cudaMalloc((void**)&d_base, array_bytes);
    if (err != cudaSuccess) {
        cleanup_cuda_memory();
        PyErr_Format(PyExc_RuntimeError, "cudaMalloc d_base failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    err = cudaMalloc((void**)&d_check, array_bytes);
    if (err != cudaSuccess) {
        cleanup_cuda_memory();
        PyErr_Format(PyExc_RuntimeError, "cudaMalloc d_check failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    err = cudaMalloc((void**)&d_values, array_bytes);
    if (err != cudaSuccess) {
        cleanup_cuda_memory();
        PyErr_Format(PyExc_RuntimeError, "cudaMalloc d_values failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    // Step 5: Copy data to GPU (synchronous)
    const char* data_ptr = raw + 12;
    
    err = cudaMemcpy(d_base, data_ptr, array_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup_cuda_memory();
        PyErr_Format(PyExc_RuntimeError, "cudaMemcpy d_base failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    err = cudaMemcpy(d_check, data_ptr + array_bytes, array_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup_cuda_memory();
        PyErr_Format(PyExc_RuntimeError, "cudaMemcpy d_check failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    err = cudaMemcpy(d_values, data_ptr + (array_bytes * 2), array_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cleanup_cuda_memory();
        PyErr_Format(PyExc_RuntimeError, "cudaMemcpy d_values failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    // Step 6: Sync and verify
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cleanup_cuda_memory();
        PyErr_Format(PyExc_RuntimeError, "cudaDeviceSynchronize failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    trie_size = sz;
    engine_loaded = true;
    
    // Return success info (use snprintf because PyUnicode_FromFormat doesn't support %f)
    char msg[256];
    snprintf(msg, sizeof(msg), "Loaded %u entries (%.2f MB) to GPU", 
             sz, (array_bytes * 3) / (1024.0 * 1024.0));
    return PyUnicode_FromString(msg);
}

// --- BATCH TOKENIZATION ---
static PyObject* tokenize_batch_gpu(PyObject* self, PyObject* args) {
    PyObject* list_obj;
    if (!PyArg_ParseTuple(args, "O", &list_obj)) return NULL;
    
    if (!PyList_Check(list_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected list of strings");
        return NULL;
    }
    
    Py_ssize_t n = PyList_Size(list_obj);
    if (n == 0) {
        return PyList_New(0);
    }
    
    // Check engine state
    if (!engine_loaded || !d_base || !d_check || !d_values) {
        PyErr_SetString(PyExc_RuntimeError, "CUDA engine not loaded. Call load_gpu() first.");
        return NULL;
    }
    
    // Build text pool and offsets
    std::vector<char> text_pool;
    std::vector<int> offsets;
    offsets.reserve(n + 1);
    
    size_t total_chars = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        PyObject* item = PyList_GetItem(list_obj, i);
        if (!PyUnicode_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "List must contain only strings");
            return NULL;
        }
        
        Py_ssize_t len;
        const char* str = PyUnicode_AsUTF8AndSize(item, &len);
        if (!str) return NULL;
        
        offsets.push_back((int)total_chars);
        text_pool.insert(text_pool.end(), str, str + len);
        total_chars += len;
    }
    offsets.push_back((int)total_chars);
    
    // Calculate maximum length of any sentence in the batch to avoid truncation
    int max_len = 0;
    for (Py_ssize_t i = 0; i < n; ++i) {
        int sentence_len = offsets[i+1] - offsets[i];
        if (sentence_len > max_len) {
            max_len = sentence_len;
        }
    }
    
    // Allocate enough space for worst-case (every char is a token)
    int max_tok = max_len + 64;
    
    // To prevent GPU OOM, cap the total elements allocated in d_out.
    // 512M elements * 4 bytes = 2 GB limit.
    long long total_elements_cap = 512LL * 1024 * 1024;
    int absolute_max_tok = (int)(total_elements_cap / n);
    if (absolute_max_tok < 1024) absolute_max_tok = 1024; // Ensure reasonable floor
    
    if (max_tok > absolute_max_tok) {
        max_tok = absolute_max_tok;
    }
    if (max_tok < 64) max_tok = 64;
    
    // Allocate GPU buffers
    char* d_text = nullptr;
    int* d_offsets = nullptr;
    int* d_out = nullptr;
    int* d_counts = nullptr;
    cudaError_t err;
    
    err = cudaMalloc((void**)&d_text, total_chars);
    if (err != cudaSuccess) {
        PyErr_Format(PyExc_RuntimeError, "cudaMalloc d_text failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    err = cudaMalloc((void**)&d_offsets, offsets.size() * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_text);
        PyErr_Format(PyExc_RuntimeError, "cudaMalloc d_offsets failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    err = cudaMalloc((void**)&d_out, n * max_tok * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_text); cudaFree(d_offsets);
        PyErr_Format(PyExc_RuntimeError, "cudaMalloc d_out failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    err = cudaMalloc((void**)&d_counts, n * sizeof(int));
    if (err != cudaSuccess) {
        cudaFree(d_text); cudaFree(d_offsets); cudaFree(d_out);
        PyErr_Format(PyExc_RuntimeError, "cudaMalloc d_counts failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    // Zero output buffers
    cudaMemset(d_out, 0, n * max_tok * sizeof(int));
    cudaMemset(d_counts, 0, n * sizeof(int));
    
    // Copy input data
    cudaMemcpy(d_text, text_pool.data(), total_chars, cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threads = 128;  // Conservative for stability
    int blocks = ((int)n + threads - 1) / threads;
    
    tokenize_kernel<<<blocks, threads>>>(
        d_base, d_check, d_values,
        d_text, d_offsets, d_out, d_counts,
        (int)n, max_tok, trie_size
    );
    
    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_text); cudaFree(d_offsets); cudaFree(d_out); cudaFree(d_counts);
        PyErr_Format(PyExc_RuntimeError, "Kernel launch failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_text); cudaFree(d_offsets); cudaFree(d_out); cudaFree(d_counts);
        PyErr_Format(PyExc_RuntimeError, "Kernel execution failed: %s", cudaGetErrorString(err));
        return NULL;
    }
    
    // Copy results back
    std::vector<int> h_out(n * max_tok);
    std::vector<int> h_counts(n);
    
    cudaMemcpy(h_out.data(), d_out, n * max_tok * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts.data(), d_counts, n * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Cleanup GPU buffers
    cudaFree(d_text);
    cudaFree(d_offsets);
    cudaFree(d_out);
    cudaFree(d_counts);
    
    // Build Python result
    PyObject* result = PyList_New(n);
    for (Py_ssize_t i = 0; i < n; ++i) {
        int count = h_counts[i];
        PyObject* tokens = PyList_New(count);
        for (int j = 0; j < count; ++j) {
            PyList_SetItem(tokens, j, PyLong_FromLong(h_out[i * max_tok + j]));
        }
        PyList_SetItem(result, i, tokens);
    }
    
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
    cleanup_cuda_memory();
}

// --- MODULE DEFINITION ---
static PyMethodDef CudaMethods[] = {
    {"load_gpu", load_gpu, METH_VARARGS, "Load DAT vocabulary to GPU memory"},
    {"tokenize_batch_gpu", tokenize_batch_gpu, METH_VARARGS, "Tokenize batch of strings on GPU"},
    {"get_hardware_info", get_hardware_info, METH_VARARGS, "Get CUDA device information"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cuda_module = {
    PyModuleDef_HEAD_INIT,
    "crayon_cuda",
    "XERV Crayon CUDA Backend v3.0 - Production Grade",
    -1,
    CudaMethods,
    NULL, NULL, NULL,
    module_cleanup
};

PyMODINIT_FUNC PyInit_crayon_cuda(void) {
    return PyModule_Create(&cuda_module);
}
