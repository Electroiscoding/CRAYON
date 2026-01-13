#ifndef CRAYON_SIMD_OPS_H
#define CRAYON_SIMD_OPS_H

#include <stddef.h>
#include <stdint.h>
#include "trie_node.h"

/**
 * @brief SIMD-optimized character search in trie node.
 * 
 * Implementation of Algorithm from[cite: 414].
 * Uses AVX2 to search child keys in parallel.
 * 
 * @param node Pointer to the TrieNode.
 * @param target_char The character to find.
 * @return Index of the child, or -1 if not found.
 */
int find_child_simd(const TrieNode* node, uint8_t target_char);

/**
 * @brief Compare up to 32 characters simultaneously using AVX2.
 * 
 * Implementation of [cite: 487].
 * 
 * @param str1 First string buffer.
 * @param str2 Second string buffer.
 * @param length Length to compare.
 * @return 0 if equal, or difference at first mismatch.
 */
int compare_strings_avx2(const char* str1, const char* str2, size_t length);

/**
 * @brief Classify 32 characters simultaneously for common types.
 * 
 * Implementation of [cite: 525].
 * Used for high-speed Unicode category detection.
 * 
 * @param chars Input character buffer.
 * @param classifications Output classification mask buffer.
 * @param count Number of characters to process.
 */
void classify_characters_avx2(const uint8_t* chars, uint8_t* classifications, size_t count);

#endif // CRAYON_SIMD_OPS_H