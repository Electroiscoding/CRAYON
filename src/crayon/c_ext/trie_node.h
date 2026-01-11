#ifndef CRAYON_TRIE_NODE_H
#define CRAYON_TRIE_NODE_H

#include <stdint.h>

// Define 64-byte alignment macro for GCC/Clang and MSVC
#if defined(_MSC_VER)
    #define ALIGN_64 __declspec(align(64))
#else
    #define ALIGN_64 __attribute__((aligned(64)))
#endif

// Forward declaration for the recursive pointer
struct TrieNode;

/**
 * @brief Memory-optimized Trie Node aligned to CPU cache lines (64 bytes).
 * * Layout designed to minimize padding and maximize cache locality [cite: 217-230].
 * Total size: 64 bytes.
 * * Structure Layout:
 * - token_id (4 bytes): The ID if this node ends a token, -1 otherwise.
 * - child_count (2 bytes): Number of children for this node.
 * - flags (2 bytes): Metadata (e.g., is_terminal, is_leaf).
 * - child_bitmap (8 bytes): Bitmap for fast ASCII child lookup (0-63).
 * - children (8 bytes): Pointer to the children array.
 * - padding (40 bytes): Used for future SIMD masks or alignment.
 */
typedef struct ALIGN_64 TrieNode {
    int32_t token_id;           // 4 bytes: Token ID (-1 if non-terminal)
    uint16_t child_count;       // 2 bytes: Number of children
    uint16_t flags;             // 2 bytes: Bit 0: is_terminal, Bit 1: has_simd_children
    uint64_t child_bitmap;      // 8 bytes: Fast lookup for first 64 ASCII chars
    
    struct TrieNode* children;  // 8 bytes (on 64-bit): Pointer to children array
    
    // Padding to reach exactly 64 bytes
    // Current usage: 4 + 2 + 2 + 8 + 8 = 24 bytes
    // Need 40 bytes padding
    uint8_t padding[40];       
    
    // In the "compact" layout described in paper, the 'children' array
    // might be stored contiguously, and 'child_chars' for SIMD would reside here.
    // For this implementation, we reserve this space for future SIMD masks.
} TrieNode;

// Compile-time check to ensure strict 64-byte size
_Static_assert(sizeof(TrieNode) == 64, "TrieNode must be exactly 64 bytes");

#endif // CRAYON_TRIE_NODE_H