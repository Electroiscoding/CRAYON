import re

with open('/home/soham/CRAYON/src/crayon/c_ext/crayon_turbo.c', 'r') as f:
    code = f.read()

# Replace the word_key function
old_word_key = r'''static inline uint64_t word_key\(const uint8_t \* restrict d, size_t n\) \{
.*?
\}'''

new_word_key = '''static inline uint64_t word_key(const uint8_t * restrict d, size_t n) {
    uint64_t h;
    if (n <= 7) {
        uint64_t v = 0; memcpy(&v, d, n);
        h = v | ((uint64_t)n << 56);
    } else if (n <= 15) {
        uint64_t v1 = 0, v2 = 0;
        memcpy(&v1, d, 8); memcpy(&v2, d + 8, n - 8);
        h = v1 ^ (v2 * 1099511628211ULL);
        h ^= (h >> 33); h *= 0xff51afd7ed558ccdULL; h ^= (h >> 33);
        h = (h & 0x00FFFFFFFFFFFFFFULL) | ((uint64_t)n << 56);
    } else {
        h = 14695981039346656037ULL;
        for (size_t i = 0; i < n; ++i) {
            h ^= (uint64_t)d[i];
            h *= 1099511628211ULL;
        }
        h ^= (h >> 33); h *= 0xff51afd7ed558ccdULL; h ^= (h >> 33);
        h = (h & 0x00FFFFFFFFFFFFFFULL) | ((uint64_t)n << 56);
    }
    return h;
}'''

code = re.sub(old_word_key, new_word_key, code, flags=re.DOTALL)

# Replace the inline hash block
old_inline = r'''                uint64_t key;
                if \(__builtin_expect\(ws \+ 16 <= len, 1\)\) \{
.*?
                WCEntry \*ce = &wc\[sl\];'''

new_inline = '''                uint64_t key;
                if (__builtin_expect(ws + 16 <= len, 1)) {
                    uint64_t v; memcpy(&v, text + ws, 8);
                    static const uint64_t M[9] = {0,0xFF,0xFFFF,0xFFFFFF,0xFFFFFFFFULL,0xFFFFFFFFFFULL,0xFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
                    if (__builtin_expect(wl <= 7, 1)) {
                        key = (v & M[wl]) | ((uint64_t)wl << 56);
                    } else if (__builtin_expect(wl <= 15, 1)) {
                        uint64_t v2; memcpy(&v2, text + ws + 8, 8);
                        v2 &= M[wl - 8];
                        uint64_t h = v ^ (v2 * 1099511628211ULL);
                        h ^= (h >> 33); h *= 0xff51afd7ed558ccdULL; h ^= (h >> 33);
                        key = (h & 0x00FFFFFFFFFFFFFFULL) | ((uint64_t)wl << 56);
                    } else {
                        key = word_key(text + ws, wl);
                    }
                } else {
                    key = word_key(text + ws, wl);
                }
                uint32_t sl = (uint32_t)((key * 11400714819323198485ULL) >> 48);
                WCEntry *ce = &wc[sl];'''

code = re.sub(old_inline, new_inline, code, flags=re.DOTALL)

with open('/home/soham/CRAYON/src/crayon/c_ext/crayon_turbo.c', 'w') as f:
    f.write(code)

