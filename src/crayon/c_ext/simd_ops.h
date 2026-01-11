#ifndef CRAYON_SIMD_OPS_H
#define CRAYON_SIMD_OPS_H

#include <stddef.h>
#include <stdint.h>
#include "trie_node.h"

/**
 * @brief Finds a child node index for a given character using AVX2 SIMD.
 * * @param child_chars Pointer to an array of characters representing child keys.
 * Must be aligned for best performance.
 * @param count Number of children to search.
 * @param target The character to search for.
 * @return Index of the character in the array, or -1 if not found.
 */
int find_child_simd(const uint8_t* child_chars, int count, uint8_t target);

/**
 * @brief Compares two strings using AVX2 256-bit vectors.
 * * Optimized for finding the longest common prefix or exact match verification.
 * * @param s1 First string buffer.
 * @param s2 Second string buffer.
 * @param len Length to compare.
 * @return 0 if equal, non-zero if different (standard memcmp semantics).
 */
int compare_strings_simd(const char* s1, const char* s2, size_t len);

/**
 * @brief Vectorized classification of characters (e.g., is_alpha, is_space).
 * * Used during normalization and pre-tokenization scanning [cite: 524-546].
 * * @param src Input string buffer.
 * @param len Length of string.
 * @param out_mask Output buffer for classification masks.
 */
void classify_chars_simd(const uint8_t* src, size_t len, uint8_t* out_mask);

#endif // CRAYON_SIMD_OPS_H