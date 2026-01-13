#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <string.h>
#include "trie_node.h"
#include "simd_ops.h"

// ----------------------------------------------------------------------------
// Builder Structures (Intermediate, non-aligned for construction)
// ----------------------------------------------------------------------------

typedef struct BuilderNode {
    int32_t token_id;
    uint8_t key;
    struct BuilderNode* first_child;
    struct BuilderNode* next_sibling;
} BuilderNode;

static BuilderNode* create_builder_node(uint8_t key) {
    BuilderNode* node = (BuilderNode*)malloc(sizeof(BuilderNode));
    if (node) {
        node->token_id = -1;
        node->key = key;
        node->first_child = NULL;
        node->next_sibling = NULL;
    }
    return node;
}

static void free_builder_node(BuilderNode* node) {
    if (!node) return;
    free_builder_node(node->first_child);
    free_builder_node(node->next_sibling);
    free(node);
}

// ----------------------------------------------------------------------------
// Trie Memory Management
// ----------------------------------------------------------------------------

/**
 * @brief Recursively frees the TrieNode structure contents.
 */
static void free_trie_node_contents(TrieNode* node) {
    if (!node) return;

    if (node->child_count > 0 && node->children) {
        for (uint16_t i = 0; i < node->child_count; i++) {
            // Recurse to free grandchildren arrays
            free_trie_node_contents(&node->children[i]);
        }
        // Free aligned arrays - use our aligned free
        free_trie_node_array(node->children);
        free(node->child_chars);
        node->children = NULL;
        node->child_chars = NULL;
    }
}

static void capsule_cleanup(PyObject* capsule) {
    TrieNode* root = (TrieNode*)PyCapsule_GetPointer(capsule, "crayon_trie_root");
    if (root) {
        free_trie_node_contents(root);
        // Free the root itself (allocated with aligned_alloc_64)
        aligned_free_64(root);
    }
}

// ----------------------------------------------------------------------------
// Builder Logic - Populate TrieNode from BuilderNode with ALIGNED allocation
// ----------------------------------------------------------------------------

static int populate_trie_node(TrieNode* t_node, BuilderNode* b_node) {
    // Clear to zeros
    memset(t_node, 0, sizeof(TrieNode));
    t_node->token_id = b_node->token_id;
    t_node->child_bitmap = 0;
    
    // Count children
    int count = 0;
    BuilderNode* curr = b_node->first_child;
    while (curr) { count++; curr = curr->next_sibling; }
    t_node->child_count = (uint16_t)count;

    if (count > 0) {
        // Allocate ALIGNED Children Array (CRITICAL for cache line optimization)
        t_node->children = alloc_trie_node_array(count);
        if (!t_node->children) return -1;
        
        // Allocate Char Array (Padding for SIMD over-read safety - round up to 32)
        int char_pad = (count + 31) & ~31;
        t_node->child_chars = (uint8_t*)calloc(char_pad, sizeof(uint8_t));
        if (!t_node->child_chars) {
            free_trie_node_array(t_node->children);
            t_node->children = NULL;
            return -1;
        }

        // Sort children by key for binary search (required for SIMD masking)
        // First collect into arrays
        BuilderNode** child_ptrs = (BuilderNode**)malloc(count * sizeof(BuilderNode*));
        if (!child_ptrs) {
            free_trie_node_array(t_node->children);
            free(t_node->child_chars);
            return -1;
        }
        
        curr = b_node->first_child;
        for (int i = 0; i < count; i++) {
            child_ptrs[i] = curr;
            curr = curr->next_sibling;
        }
        
        // Sort by key (simple insertion sort for small arrays)
        for (int i = 1; i < count; i++) {
            BuilderNode* key_node = child_ptrs[i];
            int j = i - 1;
            while (j >= 0 && child_ptrs[j]->key > key_node->key) {
                child_ptrs[j + 1] = child_ptrs[j];
                j--;
            }
            child_ptrs[j + 1] = key_node;
        }

        // Populate in sorted order
        for (int i = 0; i < count; i++) {
            BuilderNode* child_b = child_ptrs[i];
            t_node->child_chars[i] = child_b->key;
            
            // Set bitmap bit for O(1) existence check (ASCII only)
            if (child_b->key < 64) {
                t_node->child_bitmap |= (1ULL << child_b->key);
            }
            
            // Recurse to populate child (in aligned array)
            if (populate_trie_node(&t_node->children[i], child_b) != 0) {
                free(child_ptrs);
                return -1;
            }
        }
        
        free(child_ptrs);
    }
    return 0;
}

// ----------------------------------------------------------------------------
// Python Method: build_trie
// ----------------------------------------------------------------------------

static PyObject* crayon_build_trie(PyObject* self, PyObject* args) {
    PyObject* token_list;
    if (!PyArg_ParseTuple(args, "O", &token_list)) return NULL;
    if (!PyList_Check(token_list)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of strings");
        return NULL;
    }

    // 1. Build Intermediate Tree (linked-list based for easy construction)
    BuilderNode* root_b = create_builder_node(0);
    if (!root_b) {
        PyErr_NoMemory();
        return NULL;
    }
    
    Py_ssize_t num_tokens = PyList_Size(token_list);

    for (Py_ssize_t i = 0; i < num_tokens; i++) {
        PyObject* item = PyList_GetItem(token_list, i);
        const char* token = PyUnicode_AsUTF8(item);
        if (!token) { 
            free_builder_node(root_b); 
            return NULL; 
        }

        // Skip empty tokens
        if (*token == '\0') continue;

        BuilderNode* curr = root_b;
        for (const char* c = token; *c; c++) {
            uint8_t key = (uint8_t)*c;
            
            // Find existing child
            BuilderNode* child = curr->first_child;
            BuilderNode* prev = NULL;
            while (child && child->key != key) {
                prev = child;
                child = child->next_sibling;
            }

            if (!child) {
                // Create new child
                child = create_builder_node(key);
                if (!child) {
                    free_builder_node(root_b);
                    PyErr_NoMemory();
                    return NULL;
                }
                if (prev) prev->next_sibling = child;
                else curr->first_child = child;
            }
            curr = child;
        }
        // Mark end of token
        curr->token_id = (int32_t)i;
    }

    // 2. Allocate the Root (64-byte aligned)
    TrieNode* root_t = alloc_trie_node();
    if (!root_t) {
        free_builder_node(root_b);
        PyErr_NoMemory();
        return NULL;
    }

    // 3. Populate the optimized trie from builder
    if (populate_trie_node(root_t, root_b) != 0) {
        free_builder_node(root_b);
        aligned_free_64(root_t);
        PyErr_NoMemory();
        return NULL;
    }

    // 4. Cleanup Builder Tree
    free_builder_node(root_b);

    // 5. Wrap in Capsule with destructor
    return PyCapsule_New(root_t, "crayon_trie_root", capsule_cleanup);
}

// ----------------------------------------------------------------------------
// Python Method: crayon_tokenize_fast
// ----------------------------------------------------------------------------

static PyObject* crayon_tokenize_fast(PyObject* self, PyObject* args) {
    const char* text;
    Py_ssize_t text_length;
    PyObject* vocab_obj;
    int unk_token_id;

    if (!PyArg_ParseTuple(args, "s#Oi", &text, &text_length, &vocab_obj, &unk_token_id)) {
        return NULL;
    }

    TrieNode* root = (TrieNode*)PyCapsule_GetPointer(vocab_obj, "crayon_trie_root");
    if (!root) {
        PyErr_SetString(PyExc_ValueError, "Invalid Trie Capsule");
        return NULL;
    }

    // Pre-allocate result list with estimated capacity
    Py_ssize_t estimated_tokens = text_length / 4 + 1;
    PyObject* result = PyList_New(0);
    if (!result) return NULL;
    
    Py_ssize_t position = 0;

    // Hot Loop Optimization: Pre-create unk token object
    PyObject* py_unk = PyLong_FromLong(unk_token_id);
    if (!py_unk) {
        Py_DECREF(result);
        return NULL;
    }

    while (position < text_length) {
        int match_length = 0;
        int32_t token_id = -1;
        
        const TrieNode* curr = root;
        int curr_len = 0;
        
        // Max lookahead or remaining string length
        Py_ssize_t limit = text_length - position;
        const char* sub = text + position;

        for (Py_ssize_t i = 0; i < limit; i++) {
            uint8_t target = (uint8_t)sub[i];
            
            // SIMD Child Lookup [cite: 414]
            int idx = find_child_simd(curr, target);
            if (idx == -1) break;

            curr = &curr->children[idx];
            curr_len++;

            // Track longest match
            if (curr->token_id != -1) {
                token_id = curr->token_id;
                match_length = curr_len;
            }
        }

        if (match_length > 0) {
            PyObject* val = PyLong_FromLong(token_id);
            if (!val) {
                Py_DECREF(py_unk);
                Py_DECREF(result);
                return NULL;
            }
            PyList_Append(result, val);
            Py_DECREF(val);
            position += match_length;
        } else {
            // Unknown character - use pre-created object
            PyList_Append(result, py_unk);
            position += 1;
        }
    }
    
    Py_DECREF(py_unk);
    return result;
}

// ----------------------------------------------------------------------------
// Module Registration
// ----------------------------------------------------------------------------

static PyMethodDef CrayonMethods[] = {
    {"build_trie", crayon_build_trie, METH_VARARGS, "Build SIMD-optimized C-Trie from token list"},
    {"crayon_tokenize_fast", crayon_tokenize_fast, METH_VARARGS, "SIMD-accelerated tokenization"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef crayon_core_module = {
    PyModuleDef_HEAD_INIT,
    "crayon.c_ext._core",
    "High-Performance Crayon Core with AVX2 SIMD",
    -1,
    CrayonMethods
};

PyMODINIT_FUNC PyInit__core(void) {
    return PyModule_Create(&crayon_core_module);
}