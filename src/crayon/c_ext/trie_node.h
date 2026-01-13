#ifndef CRAYON_TRIE_NODE_H
#define CRAYON_TRIE_NODE_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// Strict 64-byte alignment for Cache Line Optimization [cite: 217, 230]
#if defined(_MSC_VER)
    #define ALIGN_64 __declspec(align(64))
    #include <malloc.h>
    static __inline void* aligned_alloc_64(size_t size) {
        return _aligned_malloc(size, 64);
    }
    static __inline void aligned_free_64(void* ptr) {
        _aligned_free(ptr);
    }
#else
    #define ALIGN_64 __attribute__((aligned(64)))
    static inline void* aligned_alloc_64(size_t size) {
        void* ptr = NULL;
        if (posix_memalign(&ptr, 64, size) != 0) return NULL;
        return ptr;
    }
    static inline void aligned_free_64(void* ptr) {
        free(ptr);
    }
#endif

// Forward declaration
struct TrieNode;

/**
 * @brief High-performance Trie Node aligned to CPU cache lines.
 * 
 * CRITICAL: Each TrieNode MUST be exactly 64 bytes and 64-byte aligned
 * to ensure cache line optimization.
 * 
 * Memory Layout (Aligned 64) [cite: 218-229]:
 * - token_id (4 bytes): Token ID if terminal, -1 otherwise
 * - child_count (2 bytes): Number of children
 * - flags (2 bytes): Metadata (is_terminal, etc)
 * - child_bitmap (8 bytes): Fast ASCII child existence check
 * - children (8 bytes): Pointer to aligned array of child TrieNodes
 * - child_chars (8 bytes): Pointer to array of keys (SIMD target)
 * - padding (32 bytes): Force 64-byte total
 */
typedef struct ALIGN_64 TrieNode {
    int32_t token_id;           // 4 bytes [cite: 403]
    uint16_t child_count;       // 2 bytes [cite: 404]
    uint16_t flags;             // 2 bytes [cite: 405]
    uint64_t child_bitmap;      // 8 bytes - Fast O(1) ASCII lookup
    
    struct TrieNode* children;  // 8 bytes [cite: 410] Pointer to aligned children array
    uint8_t* child_chars;       // 8 bytes [cite: 411] Characters for SIMD lookup

    // Padding: 4 + 2 + 2 + 8 + 8 + 8 = 32 bytes used. 32 bytes padding needed.
    uint8_t padding[32];
    
} TrieNode;

// Static assertion to verify 64-byte alignment
#if defined(_MSC_VER)
    static_assert(sizeof(TrieNode) == 64, "TrieNode MUST be exactly 64 bytes");
#else
    _Static_assert(sizeof(TrieNode) == 64, "TrieNode MUST be exactly 64 bytes");
#endif

/**
 * @brief Allocate an aligned array of TrieNodes.
 * 
 * CRITICAL: Regular calloc/malloc does NOT guarantee alignment for array elements.
 * We must use aligned allocation for the entire block.
 */
static inline TrieNode* alloc_trie_node_array(size_t count) {
    if (count == 0) return NULL;
    size_t size = count * sizeof(TrieNode);
    TrieNode* arr = (TrieNode*)aligned_alloc_64(size);
    if (arr) {
        memset(arr, 0, size);
    }
    return arr;
}

/**
 * @brief Allocate a single aligned TrieNode.
 */
static inline TrieNode* alloc_trie_node(void) {
    TrieNode* node = (TrieNode*)aligned_alloc_64(sizeof(TrieNode));
    if (node) {
        memset(node, 0, sizeof(TrieNode));
        node->token_id = -1;
    }
    return node;
}

/**
 * @brief Free an aligned TrieNode array.
 */
static inline void free_trie_node_array(TrieNode* arr) {
    if (arr) {
        aligned_free_64(arr);
    }
}

#endif // CRAYON_TRIE_NODE_H