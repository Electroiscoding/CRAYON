#include "simd_ops.h"
#include <immintrin.h> // Intel Intrinsics
#include <string.h>

// ----------------------------------------------------------------------------
// SIMD Child Lookup
// ----------------------------------------------------------------------------

int find_child_simd(const uint8_t* child_chars, int count, uint8_t target) {
    // If count is small, linear search is faster than setting up SIMD
    if (count < 16) {
        for (int i = 0; i < count; i++) {
            if (child_chars[i] == target) return i;
        }
        return -1;
    }

    // Broadcast target character to all 32 bytes of a YMM register
    __m256i target_vec = _mm256_set1_epi8((char)target);

    int i = 0;
    // Process in chunks of 32
    for (; i <= count - 32; i += 32) {
        // Load 32 child characters (unaligned load is safe on modern CPUs)
        __m256i children_vec = _mm256_loadu_si256((const __m256i*)(child_chars + i));
        
        // Compare: result bytes are 0xFF if equal, 0x00 otherwise
        __m256i cmp = _mm256_cmpeq_epi8(target_vec, children_vec);
        
        // Create a bitmask from the comparison result
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(cmp);
        
        if (mask != 0) {
            // Found a match! Count trailing zeros to get the index.
            // __builtin_ctz is GCC/Clang specific. MSVC uses _BitScanForward.
            return i + __builtin_ctz(mask);
        }
    }

    // Handle remaining elements scalar way
    for (; i < count; i++) {
        if (child_chars[i] == target) return i;
    }

    return -1;
}

// ----------------------------------------------------------------------------
// SIMD String Comparison
// ----------------------------------------------------------------------------

int compare_strings_simd(const char* s1, const char* s2, size_t len) {
    size_t i = 0;
    
    // Process 32-byte chunks
    for (; i + 32 <= len; i += 32) {
        __m256i v1 = _mm256_loadu_si256((const __m256i*)(s1 + i));
        __m256i v2 = _mm256_loadu_si256((const __m256i*)(s2 + i));
        
        // Compare equality
        __m256i eq = _mm256_cmpeq_epi8(v1, v2);
        
        // Move mask: if all bits are 1 (0xFFFFFFFF), strings are equal in this chunk
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(eq);
        
        if (mask != 0xFFFFFFFF) {
            // Mismatch found. The mask has 0s where bytes differ.
            // Invert mask to find 1s at mismatch positions.
            uint32_t diff_mask = ~mask;
            int offset = __builtin_ctz(diff_mask);
            // Return difference of the specific mismatching bytes
            return (unsigned char)s1[i + offset] - (unsigned char)s2[i + offset];
        }
    }
    
    // Handle remaining bytes
    return memcmp(s1 + i, s2 + i, len - i);
}

// ----------------------------------------------------------------------------
// SIMD Character Classification
// ----------------------------------------------------------------------------

void classify_chars_simd(const uint8_t* src, size_t len, uint8_t* out_mask) {
    // Constants for ranges
    const __m256i 'a' = _mm256_set1_epi8('a');
    const __m256i 'z' = _mm256_set1_epi8('z');
    const __m256i '0' = _mm256_set1_epi8('0');
    const __m256i '9' = _mm256_set1_epi8('9');
    const __m256i space = _mm256_set1_epi8(' ');

    size_t i = 0;
    for (; i + 32 <= len; i += 32) {
        __m256i chars = _mm256_loadu_si256((const __m256i*)(src + i));
        
        // Check ranges (Note: PCMPGT compares signed bytes)
        // Trick: (c >= 'a' && c <= 'z') can be done with unsigned comparisons or carefully range shifting
        // Here we use standard signed comparison logic for simplicity
        
        // Is Lower Alpha: chars >= 'a' AND chars <= 'z'
        // (Implementation omitted for brevity, usually involves sub/add wrapper)
        // ...
        
        // Placeholder for the "is_space" logic which is simplest:
        __m256i is_sp = _mm256_cmpeq_epi8(chars, space);
        
        // Store 0xFF where space, 0x00 otherwise directly to output mask
        // In reality we pack bits, but here we write byte-masks for simplicity
        _mm256_storeu_si256((__m256i*)(out_mask + i), is_sp);
    }
    
    // Scalar fallback
    for (; i < len; i++) {
        out_mask[i] = (src[i] == ' ') ? 0xFF : 0x00;
    }
}