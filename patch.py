import re

with open('/home/soham/CRAYON/src/crayon/c_ext/crayon_turbo.c', 'r') as f:
    code = f.read()

# Replace the word_key call block in tokenize_one
old_block = """                uint64_t key = word_key(text + ws, wl);
                uint32_t sl = (uint32_t)(key & WORD_CACHE_MASK);
                WCEntry *ce = &wc[sl];"""

new_block = """                uint64_t key;
                if (__builtin_expect(ws + 16 <= len, 1)) {
                    uint64_t v; memcpy(&v, text + ws, 8);
                    if (__builtin_expect(wl <= 8, 1)) {
                        static const uint64_t M[9] = {0,0xFF,0xFFFF,0xFFFFFF,0xFFFFFFFFULL,0xFFFFFFFFFFULL,0xFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
                        v &= M[wl];
                        v ^= (v >> 33); v *= 0xff51afd7ed558ccdULL; v ^= (v >> 33);
                        key = (v & 0x00FFFFFFFFFFFFFFULL) | ((uint64_t)wl << 56);
                    } else if (__builtin_expect(wl <= 16, 1)) {
                        uint64_t v2; memcpy(&v2, text + ws + 8, 8);
                        static const uint64_t M[9] = {0,0xFF,0xFFFF,0xFFFFFF,0xFFFFFFFFULL,0xFFFFFFFFFFULL,0xFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
                        v2 &= M[wl - 8];
                        uint64_t h = 14695981039346656037ULL;
                        h ^= v; h *= 1099511628211ULL;
                        h ^= v2; h *= 1099511628211ULL;
                        h ^= (h >> 33); h *= 0xff51afd7ed558ccdULL; h ^= (h >> 33);
                        key = (h & 0x00FFFFFFFFFFFFFFULL) | ((uint64_t)wl << 56);
                    } else {
                        key = word_key(text + ws, wl);
                    }
                } else {
                    key = word_key(text + ws, wl);
                }
                uint32_t sl = (uint32_t)(key & WORD_CACHE_MASK);
                WCEntry *ce = &wc[sl];"""

code = code.replace(old_block, new_block)

with open('/home/soham/CRAYON/src/crayon/c_ext/crayon_turbo.c', 'w') as f:
    f.write(code)

