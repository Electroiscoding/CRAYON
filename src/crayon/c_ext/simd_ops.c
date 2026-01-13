#include "simd_ops.h"
#include <immintrin.h>
#include <string.h>

// Cross-platform count trailing zeros (CTZ) macro
#if defined(_MSC_VER)
    #include <intrin.h>
    static __inline int ctz32(uint32_t value) {
        unsigned long index;
        _BitScanForward(&index, value);
        return (int)index;
    }
    #define CTZ(x) ctz32(x)
#else
    #define CTZ(x) __builtin_ctz(x)
#endif

// Helper for binary search fallback [cite: 426]
static inline int binary_search_chars(const uint8_t* chars, int count, uint8_t target) {
    int left = 0, right = count - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (chars[mid] == target) return mid;
        if (chars[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// [cite: 414] SIMD-optimized character search
int find_child_simd(const TrieNode* node, uint8_t target_char) {
    // Handle empty nodes (leaf nodes with no children)
    if (node->child_count == 0 || node->child_chars == NULL) {
        return -1;
    }
    
    // [cite: 415] Use SIMD for small child sets (<= 16)
    if (node->child_count <= 16) {
        // [cite: 418] Set target vector
        __m128i target_vec = _mm_set1_epi8((char)target_char);
        
        // Load child characters (unaligned load is safe)
        // Note: child_chars must be padded to 16 bytes allocation-side
        __m128i chars_vec = _mm_loadu_si128((__m128i*)node->child_chars);
        
        // [cite: 420] Compare
        __m128i cmp_result = _mm_cmpeq_epi8(target_vec, chars_vec);
        
        // [cite: 421] Create mask
        int mask = _mm_movemask_epi8(cmp_result);
        
        // Mask out positions beyond child_count
        mask &= (1 << node->child_count) - 1;
        
        // [cite: 422] Check result
        if (mask == 0) return -1;
        
        // [cite: 423] Return index of first match (Count Trailing Zeros)
        return CTZ((uint32_t)mask);
    } else {
        // [cite: 425] Fallback to binary search for large child sets
        return binary_search_chars(node->child_chars, node->child_count, target_char);
    }
}

// [cite: 487] Compare strings using AVX2
int compare_strings_avx2(const char* str1, const char* str2, size_t length) {
    size_t i = 0;
    
    // [cite: 489] Process in 32-byte chunks
    for (; i + 32 <= length; i += 32) {
        // Load 256-bit vectors
        __m256i vec1 = _mm256_loadu_si256((const __m256i*)(str1 + i));
        __m256i vec2 = _mm256_loadu_si256((const __m256i*)(str2 + i));
        
        // [cite: 493] Compare equality
        __m256i cmp = _mm256_cmpeq_epi8(vec1, vec2);
        
        // [cite: 495] Move mask
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(cmp);
        
        // [cite: 496] If not all ones (0xFFFFFFFF), we found a mismatch
        if (mask != 0xFFFFFFFF) {
            // [cite: 498] Find exact position
            int offset = CTZ(~mask);
            return (unsigned char)str1[i + offset] - (unsigned char)str2[i + offset];
        }
    }
    
    // [cite: 502] Handle remaining bytes
    for (; i < length; i++) {
        if (str1[i] != str2[i]) {
            return (unsigned char)str1[i] - (unsigned char)str2[i];
        }
    }
    
    // [cite: 505] Strings match
    return 0;
}

// [cite: 525] Vectorized Character Classification
void classify_characters_avx2(const uint8_t* chars, uint8_t* classifications, size_t count) {
    // [cite: 526-529] Pre-computed constants
    const __m256i alpha_min = _mm256_set1_epi8('a');
    const __m256i alpha_max = _mm256_set1_epi8('z');
    const __m256i digit_min = _mm256_set1_epi8('0');
    const __m256i digit_max = _mm256_set1_epi8('9');
    const __m256i space_char = _mm256_set1_epi8(' ');
    
    size_t i = 0;
    // [cite: 530] Loop 32 chars at a time
    for (; i + 32 <= count; i += 32) {
        // [cite: 532] Load
        __m256i char_vec = _mm256_loadu_si256((const __m256i*)(chars + i));
        
        // [cite: 533-536] Is Alpha logic (simplified for AVX comparison quirks)
        // Note: PCMPGT compares signed bytes. We assume ASCII range here.
        __m256i is_alpha = _mm256_and_si256(
            _mm256_cmpgt_epi8(char_vec, _mm256_sub_epi8(alpha_min, _mm256_set1_epi8(1))),
            _mm256_cmpgt_epi8(_mm256_add_epi8(alpha_max, _mm256_set1_epi8(1)), char_vec)
        );

        // [cite: 537-539] Is Digit logic
        __m256i is_digit = _mm256_and_si256(
            _mm256_cmpgt_epi8(char_vec, _mm256_sub_epi8(digit_min, _mm256_set1_epi8(1))),
            _mm256_cmpgt_epi8(_mm256_add_epi8(digit_max, _mm256_set1_epi8(1)), char_vec)
        );
        
        // [cite: 540] Is Space
        __m256i is_space = _mm256_cmpeq_epi8(char_vec, space_char);
        
        // [cite: 543-544] Combine results: Alpha=1, Digit=2, Space=4
        __m256i result = _mm256_or_si256(
            _mm256_and_si256(is_alpha, _mm256_set1_epi8(1)),
            _mm256_or_si256(
                _mm256_and_si256(is_digit, _mm256_set1_epi8(2)),
                _mm256_and_si256(is_space, _mm256_set1_epi8(4))
            )
        );
        
        // [cite: 546] Store
        _mm256_storeu_si256((__m256i*)(classifications + i), result);
    }
    
    // Fallback for remaining
    for (; i < count; i++) {
        uint8_t c = chars[i];
        classifications[i] = 0;
        if (c >= 'a' && c <= 'z') classifications[i] |= 1;
        if (c >= '0' && c <= '9') classifications[i] |= 2;
        if (c == ' ') classifications[i] |= 4;
    }
}