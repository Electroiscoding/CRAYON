#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// ----------------------------------------------------------------------------
// Double-Array Trie State (Global / Per Capsule)
// ----------------------------------------------------------------------------

typedef struct {
    int32_t* base;
    int32_t* check;
    int32_t* terminals;
    int32_t size;
    void* memory_block; // Pointer to full block to free
} DATModel;

static void dat_capsule_cleanup(PyObject* capsule) {
    DATModel* model = (DATModel*)PyCapsule_GetPointer(capsule, "crayon_dat");
    if (model) {
        if (model->memory_block) {
            free(model->memory_block);
        }
        free(model);
    }
}

// ----------------------------------------------------------------------------
// Load DAT File (.dat) - Zero-Copyish (Single Read)
// ----------------------------------------------------------------------------

static PyObject* load_dat_file(PyObject* self, PyObject* args) {
    const char* path;
    if (!PyArg_ParseTuple(args, "s", &path)) return NULL;

    FILE* f = fopen(path, "rb");
    if (!f) {
        PyErr_SetString(PyExc_IOError, "Cannot open DAT file");
        return NULL;
    }

    // Header Check
    char magic[4];
    uint32_t version;
    uint32_t size;
    
    if (fread(magic, 1, 4, f) != 4 || 
        fread(&version, 4, 1, f) != 1 || 
        fread(&size, 4, 1, f) != 1) {
        fclose(f);
        PyErr_SetString(PyExc_ValueError, "Invalid DAT header");
        return NULL;
    }

    if (memcmp(magic, "CRYN", 4) != 0) {
        fclose(f);
        PyErr_SetString(PyExc_ValueError, "Invalid Magic Bytes");
        return NULL;
    }

    // Allocate memory for the 3 arrays
    // Layout: [BASE: size*4] [CHECK: size*4] [TERM: size*4]
    size_t array_bytes = size * sizeof(int32_t);
    size_t total_bytes = array_bytes * 3;
    
    void* block = malloc(total_bytes);
    if (!block) {
        fclose(f);
        PyErr_NoMemory();
        return NULL;
    }

    if (fread(block, 1, total_bytes, f) != total_bytes) {
        free(block);
        fclose(f);
        PyErr_SetString(PyExc_IOError, "Unexpected EOF reading DAT body");
        return NULL;
    }
    
    fclose(f);

    // Setup Model Struct
    DATModel* model = (DATModel*)malloc(sizeof(DATModel));
    if (!model) {
        free(block);
        PyErr_NoMemory();
        return NULL;
    }

    model->memory_block = block;
    model->size = (int32_t)size;
    
    // Assign pointers
    char* ptr = (char*)block;
    model->base = (int32_t*)ptr;
    model->check = (int32_t*)(ptr + array_bytes);
    model->terminals = (int32_t*)(ptr + array_bytes * 2);

    return PyCapsule_New(model, "crayon_dat", dat_capsule_cleanup);
}

// ----------------------------------------------------------------------------
// Fast Tokenization (Double-Array Traversal)
// ----------------------------------------------------------------------------

static PyObject* crayon_tokenize_fast(PyObject* self, PyObject* args) {
    const char* text;
    Py_ssize_t text_length;
    PyObject* dat_capsule;
    int unk_token_id;

    if (!PyArg_ParseTuple(args, "s#Oi", &text, &text_length, &dat_capsule, &unk_token_id)) {
        return NULL;
    }

    DATModel* model = (DATModel*)PyCapsule_GetPointer(dat_capsule, "crayon_dat");
    if (!model) {
        PyErr_SetString(PyExc_ValueError, "Invalid DAT Capsule");
        return NULL;
    }

    int32_t* base = model->base;
    int32_t* check = model->check;
    int32_t* terminals = model->terminals;
    int32_t size = model->size;

    PyObject* result = PyList_New(0);
    if (!result) return NULL;

    PyObject* py_unk = PyLong_FromLong(unk_token_id);
    if (!py_unk) {
        Py_DECREF(result);
        return NULL;
    }

    Py_ssize_t position = 0;
    while (position < text_length) {
        // DAT Traversal
        // Algorithm:
        // s = 0 (root)
        // for c in text:
        //   t = base[s] + c
        //   if check[t] == s:
        //      s = t
        //      if terminals[s] != -1: match
        //   else: break
        
        int s = 0; // Root state
        int32_t best_token = -1;
        int best_len = 0;

        for (Py_ssize_t i = 0; position + i < text_length; i++) {
            uint8_t c = (uint8_t)text[position + i];
            
            // Bounds check not strictly needed if base array logic is standard,
            // but necessary to prevent OOB read if base[s] is large.
            // Check if transition is valid
            if (s >= size) break;
            
            int offset = base[s] + c;
            
            if (offset >= size || offset < 0) {
                 break; // Invalid
            }
            
            if (check[offset] != s) {
                break; // Mismatch
            }
            
            // Move to next state
            s = offset;
            
            // Is it a word end?
            if (terminals[s] != -1) {
                best_token = terminals[s];
                best_len = (int)(i + 1);
            }
        }

        if (best_len > 0) {
            PyObject* val = PyLong_FromLong(best_token);
            if (!val) {
                Py_DECREF(result);
                Py_DECREF(py_unk);
                return NULL;
            }
            PyList_Append(result, val);
            Py_DECREF(val);
            position += best_len;
        } else {
            // UNK
            PyList_Append(result, py_unk);
            position += 1;
        }
    }

    Py_DECREF(py_unk);
    return result;
}

// ----------------------------------------------------------------------------
// Module definition
// ----------------------------------------------------------------------------

static PyMethodDef CrayonMethods[] = {
    {"load_dat_file", load_dat_file, METH_VARARGS, "Load binary DAT file into memory"},
    {"crayon_tokenize_fast", crayon_tokenize_fast, METH_VARARGS, "Double-Array Trie Inference"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef crayon_core_module = {
    PyModuleDef_HEAD_INIT,
    "crayon.c_ext._core",
    "High-Performance DAT Engine",
    -1,
    CrayonMethods
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&crayon_core_module);
}