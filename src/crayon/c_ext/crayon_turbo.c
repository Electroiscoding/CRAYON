/*
 * CRAYON TURBO ENGINE v5.7.4 (L2-Resident Cache + Clean OMP)
 * ==============================================================================
 * Target: >= 100M tokens/sec on all real-world text, all sizes.
 *
 * Architecture:
 *  1. BYTE LUT + PAIR LUT: O(1) for single-char and 2-char tokens.
 *  2. UNIFIED WORD SCANNER: AVX2 _mm256_movemask_epi8 boundary detection.
 *  3. L2-RESIDENT 2-WAY SET-ASSOCIATIVE WORD CACHE:
 *     - 4K sets × 2 ways = 8K entries × 32B = 256KB total.
 *     - Fits ENTIRELY in L2 cache (256KB). Access = ~12 cycles, not ~40 (L3).
 *     - 8K entries with Zipf word frequency → ~95% hit rate.
 *     - Cache miss falls through to DAT trie (only ~5% of lookups).
 *  4. OMP: 8KB threshold, max_t×4 chunks with pre-sized merge.
 *  5. NUMPY OUTPUT: zero per-token Python allocation.
 *  6. PRE-BAKED INT CACHE: O(1) list output.
 */

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <numpy/arrayobject.h>

#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
  #include <omp.h>
#endif

/* ── SIMD ────────────────────────────────────────────────────── */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__)
  #define HAVE_X86 1
  #ifndef _MSC_VER
    #include <cpuid.h>
  #else
    #include <intrin.h>
  #endif
  #ifdef __AVX2__
    #include <immintrin.h>
    #define HAVE_AVX2 1
  #else
    #define HAVE_AVX2 0
  #endif
#else
  #define HAVE_X86 0
  #define HAVE_AVX2 0
#endif

/* ══════════════════════════════════════════════════════════════
 *  L2-RESIDENT 2-Way Set-Associative Word Cache
 *  ─────────────────────────────────────────────
 *  4K sets × 2 ways × 32 bytes/entry = 256KB total.
 *  This fits ENTIRELY in the L2 cache (256KB per core).
 *  L2 access = ~12 cycles vs L3 = ~40 cycles = 3.3x faster.
 *
 *  With Zipf word frequency distribution, top 8K words cover
 *  ~95% of all word occurrences → 95% L2 hit rate.
 * ══════════════════════════════════════════════════════════════ */
#define WC_SETS_SHIFT   12
#define WC_SETS         (1u << WC_SETS_SHIFT)   /* 4K sets */
#define WC_SET_MASK     (WC_SETS - 1)
#define WC_WAYS         2
#define WC_TOTAL        (WC_SETS * WC_WAYS)     /* 8K entries = 256KB */

typedef struct {
    uint64_t key;       /* hash with len in top 8 bits; 0 = unused */
    int32_t  ids[4];    /* up to 4 token IDs; unused = -1 */
    int32_t  ntoks;     /* 0 = unused/uncacheable, 1-4 = valid count */
    int32_t  _pad;      /* pad to exactly 32 bytes */
} WCEntry;  /* 32 bytes. 2 entries = 64 bytes = 1 cache line. */

/* ══════════════════════════════════════════════════════════════
 *  Global DAT State
 * ══════════════════════════════════════════════════════════════ */
#define MAX_WORD_LEN    128
#define UNK_ID          1
#define INIT_OUT_CAP    (1 << 18)

static const int32_t *g_base   = NULL;
static const int32_t *g_check  = NULL;
static const int32_t *g_values = NULL;
static uint32_t       g_size   = 0;

static int32_t g_byte_lut[256];
static int32_t g_pair_lut[65536];

static Py_buffer g_pybuf;
static int       g_pybuf_held = 0;
static PyObject *g_buf_owner  = NULL;

/* ── Pre-baked int cache ────────────────────────────────────── */
static PyObject **g_intcache    = NULL;
static uint32_t   g_intcache_sz = 0;

static void _free_intcache(void) {
    if (g_intcache) {
        for (uint32_t i = 0; i < g_intcache_sz; ++i)
            Py_XDECREF(g_intcache[i]);
        free(g_intcache);
        g_intcache = NULL; g_intcache_sz = 0;
    }
}

static int _build_intcache(uint32_t vsz) {
    _free_intcache();
    g_intcache = (PyObject **)malloc((size_t)vsz * sizeof(PyObject *));
    if (!g_intcache) return -1;
    for (uint32_t i = 0; i < vsz; ++i) {
        g_intcache[i] = PyLong_FromLong((long)i);
        if (!g_intcache[i]) {
            for (uint32_t j = 0; j < i; ++j) Py_DECREF(g_intcache[j]);
            free(g_intcache); g_intcache = NULL; g_intcache_sz = 0;
            return -1;
        }
    }
    g_intcache_sz = vsz;
    return 0;
}

static void _build_byte_lut(void) {
    for (int c = 0; c < 256; ++c) {
        int32_t nx = g_base[0] + c;
        g_byte_lut[c] = ((uint32_t)nx < g_size && g_check[nx] == 0 && g_values[nx] != -1)
                        ? g_values[nx] : -1;
    }
}

static void _build_pair_lut(void) {
    for (int c0 = 0; c0 < 256; ++c0) {
        int32_t nx1 = g_base[0] + c0;
        if ((uint32_t)nx1 < g_size && g_check[nx1] == 0) {
            for (int c1 = 0; c1 < 256; ++c1) {
                int32_t nx2 = g_base[nx1] + c1;
                if ((uint32_t)nx2 < g_size && g_check[nx2] == nx1 && g_values[nx2] != -1)
                    g_pair_lut[(c0 << 8) | c1] = g_values[nx2];
                else
                    g_pair_lut[(c0 << 8) | c1] = -1;
            }
        } else {
            for (int c1 = 0; c1 < 256; ++c1)
                g_pair_lut[(c0 << 8) | c1] = -1;
        }
    }
}

/* ── Per-thread word cache management ────────────────────── */
#define MAX_PAR_THREADS 16
static WCEntry *g_par_wc[MAX_PAR_THREADS];

static void _init_wc(WCEntry *wc) {
    memset(wc, 0, WC_TOTAL * sizeof(WCEntry));
}

static void _ensure_par_caches(void) {
    for (int i = 0; i < MAX_PAR_THREADS; ++i) {
        if (!g_par_wc[i]) {
            g_par_wc[i] = (WCEntry *)calloc(WC_TOTAL, sizeof(WCEntry));
        }
    }
}

static __thread WCEntry *g_tl_wc   = NULL;
static __thread int      g_tl_init = 0;

static WCEntry *_get_tl_wc(void) {
    if (__builtin_expect(!g_tl_init, 0)) {
        g_tl_wc = (WCEntry *)calloc(WC_TOTAL, sizeof(WCEntry));
        g_tl_init = 1;
    }
    return g_tl_wc;
}

static void _clear_par_caches(void) {
    for (int i = 0; i < MAX_PAR_THREADS; ++i) {
        if (g_par_wc[i]) _init_wc(g_par_wc[i]);
    }
    if (g_tl_wc) _init_wc(g_tl_wc);
    g_tl_init = (g_tl_wc != NULL);
}

/* ══════════════════════════════════════════════════════════════
 *  Hash Function
 * ══════════════════════════════════════════════════════════════ */
static uint8_t g_isword[256];

static void _build_isword(void) {
    for (int c = 0; c < 256; ++c)
        g_isword[c] = ((c>='a'&&c<='z')||(c>='A'&&c<='Z')||
                       (c>='0'&&c<='9')||c=='_'||c>=0x80) ? 1 : 0;
}

static inline uint64_t word_key(const uint8_t * restrict d, size_t n) {
    uint64_t h;
    if (n <= 7) {
        uint64_t v = 0; memcpy(&v, d, n);
        h = v | ((uint64_t)n << 56);
    } else if (n <= 15) {
        uint64_t v1, v2 = 0;
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
}

/* ══════════════════════════════════════════════════════════════
 *  AVX2 Boundary Scanners
 * ══════════════════════════════════════════════════════════════ */
#if HAVE_AVX2

static inline int _wc_mask32(__m256i c) {
    __m256i wu = _mm256_and_si256(_mm256_cmpgt_epi8(c, _mm256_set1_epi8('A'-1)),
                                   _mm256_cmpgt_epi8(_mm256_set1_epi8('Z'+1), c));
    __m256i wl = _mm256_and_si256(_mm256_cmpgt_epi8(c, _mm256_set1_epi8('a'-1)),
                                   _mm256_cmpgt_epi8(_mm256_set1_epi8('z'+1), c));
    __m256i wd = _mm256_and_si256(_mm256_cmpgt_epi8(c, _mm256_set1_epi8('0'-1)),
                                   _mm256_cmpgt_epi8(_mm256_set1_epi8('9'+1), c));
    __m256i ws = _mm256_cmpeq_epi8(c, _mm256_set1_epi8('_'));
    __m256i wh = _mm256_cmpgt_epi8(_mm256_setzero_si256(), c);
    return _mm256_movemask_epi8(_mm256_or_si256(
        _mm256_or_si256(_mm256_or_si256(wu, wl), _mm256_or_si256(wd, ws)), wh));
}

static size_t find_nonword(const uint8_t *t, size_t p, size_t n) {
    while (p + 32 <= n) {
        int b = _wc_mask32(_mm256_loadu_si256((const __m256i*)(t+p)));
        if (b != -1) return p + (size_t)__builtin_ctz(~b);
        p += 32;
    }
    const uint8_t *iw = g_isword;
    for (; p < n; ++p) { if (!iw[t[p]]) break; }
    return p;
}
static size_t find_word(const uint8_t *t, size_t p, size_t n) {
    while (p + 32 <= n) {
        int b = _wc_mask32(_mm256_loadu_si256((const __m256i*)(t+p)));
        if (b != 0) return p + (size_t)__builtin_ctz(b);
        p += 32;
    }
    const uint8_t *iw = g_isword;
    for (; p < n; ++p) { if (iw[t[p]]) break; }
    return p;
}

#else /* scalar */

static size_t find_nonword(const uint8_t *t, size_t p, size_t n) {
    const uint8_t *iw = g_isword;
    for (; p < n; ++p) { if (!iw[t[p]]) break; }
    return p;
}
static size_t find_word(const uint8_t *t, size_t p, size_t n) {
    const uint8_t *iw = g_isword;
    for (; p < n; ++p) { if (iw[t[p]]) break; }
    return p;
}

#endif

/* ══════════════════════════════════════════════════════════════
 *  Growable output buffer
 * ══════════════════════════════════════════════════════════════ */
typedef struct { int32_t *d; size_t n, cap; } TBuf;

static inline void tb_init(TBuf *b, size_t cap) {
    size_t alloc = (cap > 64) ? cap : INIT_OUT_CAP;
    b->d   = (int32_t *)malloc(alloc * sizeof(int32_t));
    b->n   = 0;
    b->cap = b->d ? alloc : 0;
}
static inline void tb_free(TBuf *b) { free(b->d); b->d=NULL; b->n=b->cap=0; }

/* ══════════════════════════════════════════════════════════════
 *  DAT Longest-Match
 * ══════════════════════════════════════════════════════════════ */
static inline int dat_match(const uint8_t * restrict t, size_t end, size_t pos,
                             int32_t * restrict out) {
    const int32_t *base=g_base, *check=g_check, *values=g_values;
    uint32_t sz=g_size;
    int32_t node=0, best=-1; int blen=0;

    if (__builtin_expect(pos < end, 1)) {
        int32_t nx = base[0] + t[pos];
        if (__builtin_expect((uint32_t)nx < sz && check[nx] == 0, 1)) {
            node = nx;
            int32_t v = values[node];
            if (v != -1) { best = v; blen = 1; }

            for (size_t i = pos + 1; i < end; ++i) {
                nx = base[node] + t[i];
                if (__builtin_expect((uint32_t)nx >= sz || check[nx] != node, 0)) break;
                node = nx;
                v = values[node];
                if (v != -1) { best = v; blen = (int)(i - pos) + 1; }
            }
        }
    }
    *out = best; return blen;
}

/* ══════════════════════════════════════════════════════════════
 *  Core Tokenize Loop  (v5.7.4 — L2-resident cache, no prefetch overhead)
 * ══════════════════════════════════════════════════════════════ */
static void tokenize_one(const uint8_t * restrict text, size_t len,
                          TBuf * restrict out, WCEntry * restrict wc) {
    size_t pos = 0;
    const int32_t * restrict blut = g_byte_lut;
    const uint8_t * restrict isw_lut = g_isword;

    int32_t * restrict obase = out->d;
    size_t ocap = out->cap;
    int32_t * restrict op = obase + out->n;

#define FAST_PUSH(val) do { \
        size_t on = (size_t)(op - obase); \
        if (__builtin_expect(on >= ocap, 0)) { \
            ocap = ocap ? ocap * 2 : 1024; \
            obase = (int32_t*)realloc(obase, ocap * 4); \
            op = obase + on; \
        } \
        *op++ = (val); \
    } while(0)

#define FUSE_SEPARATOR \
    while (__builtin_expect(pos < len, 1)) { \
        uint8_t _nc = text[pos]; \
        if (isw_lut[_nc]) break; \
        int32_t _lt = blut[_nc]; \
        if (__builtin_expect(_lt >= 0, 1)) { FAST_PUSH(_lt); pos++; } \
        else break; \
    }

    while (pos < len) {
        uint8_t c0 = text[pos];
        int isw = isw_lut[c0];

        if (isw) {
            /* ── Word span ── */
            size_t ws = pos;
            size_t we = find_nonword(text, ws + 1, len);
            size_t wl = we - ws;
            pos = we;

            if (__builtin_expect(wl <= MAX_WORD_LEN, 1)) {
                /* 2-char fast path via pair LUT */
                if (__builtin_expect(wl == 2, 0)) {
                    uint16_t pair = ((uint16_t)text[ws] << 8) | (uint8_t)text[ws + 1];
                    int32_t pt = g_pair_lut[pair];
                    if (__builtin_expect(pt >= 0, 1)) {
                        FAST_PUSH(pt);
                        FUSE_SEPARATOR
                        continue;
                    }
                }

                /* Compute hash */
                uint64_t key;
                if (__builtin_expect(ws + 16 <= len, 1)) {
                    uint64_t v; memcpy(&v, text + ws, 8);
                    static const uint64_t M[9] = {0,0xFF,0xFFFF,0xFFFFFF,0xFFFFFFFFULL,
                        0xFFFFFFFFFFULL,0xFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFULL,0xFFFFFFFFFFFFFFFFULL};
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

                /* 2-way set-associative lookup (entire set in 1 cache line) */
                uint32_t si = (uint32_t)((key * 11400714819323198485ULL) >> 52) & WC_SET_MASK;
                WCEntry *e0 = &wc[si * 2];
                WCEntry *e1 = e0 + 1;

                /* Check way 0 */
                if (__builtin_expect(e0->key == key, 1)) {
                    if (__builtin_expect(e0->ntoks > 0, 1)) {
                        int nt = e0->ntoks;
                        FAST_PUSH(e0->ids[0]);
                        if (nt >= 2) FAST_PUSH(e0->ids[1]);
                        if (nt >= 3) FAST_PUSH(e0->ids[2]);
                        if (nt == 4) FAST_PUSH(e0->ids[3]);
                        FUSE_SEPARATOR
                        continue;
                    }
                    goto slow_word;
                }
                /* Check way 1 */
                if (__builtin_expect(e1->key == key, 0)) {
                    if (__builtin_expect(e1->ntoks > 0, 1)) {
                        int nt = e1->ntoks;
                        FAST_PUSH(e1->ids[0]);
                        if (nt >= 2) FAST_PUSH(e1->ids[1]);
                        if (nt >= 3) FAST_PUSH(e1->ids[2]);
                        if (nt == 4) FAST_PUSH(e1->ids[3]);
                        /* Promote to way 0 (swap) */
                        WCEntry tmp = *e0; *e0 = *e1; *e1 = tmp;
                        FUSE_SEPARATOR
                        continue;
                    }
                    goto slow_word;
                }

                /* ── CACHE MISS ── */
slow_word:;
                {
                    size_t wp = ws;
                    int32_t ts[4] = {-1, -1, -1, -1}; int cnt = 0;
                    while (wp < we) {
                        int32_t tid; int ml = dat_match(text, we, wp, &tid);
                        if (ml > 0) {
                            FAST_PUSH(tid);
                            if (cnt < 4) ts[cnt] = tid;
                            cnt++; wp += ml;
                        } else {
                            FAST_PUSH(UNK_ID);
                            if (cnt < 4) ts[cnt] = UNK_ID;
                            cnt++; wp++;
                        }
                    }
                    /* Insert: evict way 1, promote to way 0 */
                    *e1 = *e0;
                    e0->key = key;
                    if (cnt >= 1 && cnt <= 4) {
                        e0->ntoks = cnt;
                        e0->ids[0] = ts[0];
                        e0->ids[1] = (cnt >= 2) ? ts[1] : -1;
                        e0->ids[2] = (cnt >= 3) ? ts[2] : -1;
                        e0->ids[3] = (cnt == 4) ? ts[3] : -1;
                        FUSE_SEPARATOR
                    } else {
                        e0->ntoks = 0;
                        e0->ids[0] = -1; e0->ids[1] = -1;
                        e0->ids[2] = -1; e0->ids[3] = -1;
                    }
                }
            } else {
                /* Word > MAX_WORD_LEN: raw DAT */
                size_t wp = ws;
                while (wp < we) {
                    int32_t tid; int ml = dat_match(text, we, wp, &tid);
                    if (ml > 0) { FAST_PUSH(tid); wp += ml; }
                    else        { FAST_PUSH(UNK_ID); wp++; }
                }
            }
        } else {
            /* ── Non-word run ── */
            if (__builtin_expect(pos + 1 < len, 1)) {
                uint8_t c1 = text[pos + 1];
                if (!isw_lut[c1]) {
                    uint16_t pair = ((uint16_t)c0 << 8) | c1;
                    int32_t pt = g_pair_lut[pair];
                    if (__builtin_expect(pt >= 0, 1)) {
                        FAST_PUSH(pt);
                        pos += 2;
                        continue;
                    }
                }
            }
            int32_t lt0 = blut[c0];
            pos++;
            if (__builtin_expect(lt0 >= 0, 1)) {
                FAST_PUSH(lt0);
                if (__builtin_expect(pos < len, 1)) {
                    if (isw_lut[text[pos]]) continue;
                }
                while (pos < len) {
                    uint8_t cx = text[pos];
                    if (isw_lut[cx]) break;
                    int32_t ltx = blut[cx];
                    if (__builtin_expect(ltx >= 0, 1)) { FAST_PUSH(ltx); pos++; }
                    else {
                        size_t ne = find_word(text, pos, len);
                        for (size_t wp=pos; wp<ne; ) {
                            ltx = blut[(uint8_t)text[wp]];
                            if (__builtin_expect(ltx>=0,1)) { FAST_PUSH(ltx); wp++; }
                            else {
                                int32_t tid; int ml=dat_match(text,ne,wp,&tid);
                                if (ml>0) { FAST_PUSH(tid); wp+=ml; }
                                else      { FAST_PUSH(UNK_ID); wp++; }
                            }
                        }
                        pos = ne; break;
                    }
                }
            } else {
                size_t ne = find_word(text, pos-1, len);
                for (size_t wp=pos-1; wp<ne; ) {
                    int32_t lt = blut[(uint8_t)text[wp]];
                    if (__builtin_expect(lt>=0,1)) { FAST_PUSH(lt); wp++; }
                    else {
                        int32_t tid; int ml=dat_match(text,ne,wp,&tid);
                        if (ml>0) { FAST_PUSH(tid); wp+=ml; }
                        else      { FAST_PUSH(UNK_ID); wp++; }
                    }
                }
                pos = ne;
            }
        }
    }
#undef FUSE_SEPARATOR
#undef FAST_PUSH
    out->d = obase;
    out->cap = ocap;
    out->n = (size_t)(op - obase);
}

/* ══════════════════════════════════════════════════════════════
 *  Output Converters
 * ══════════════════════════════════════════════════════════════ */
static PyObject *tb_to_numpy(const TBuf *b) {
    npy_intp dims[1]={(npy_intp)b->n};
    PyObject *a=PyArray_SimpleNew(1,dims,NPY_INT32);
    if (!a) return NULL;
    if (b->n) memcpy(PyArray_DATA((PyArrayObject*)a),b->d,b->n*4);
    return a;
}
static PyObject *tb_to_pylist(const TBuf *b) {
    Py_ssize_t n=(Py_ssize_t)b->n;
    PyObject *lst=PyList_New(n); if (!lst) return NULL;
    const int32_t *ids=b->d; uint32_t csz=g_intcache_sz;
    PyObject **ic=g_intcache;
    for (Py_ssize_t i=0;i<n;++i) {
        int32_t id=ids[i]; PyObject *v;
        if (__builtin_expect((uint32_t)id<csz,1)) { v=ic[id]; Py_INCREF(v); }
        else { v=PyLong_FromLong((long)id); if (!v) { Py_DECREF(lst); return NULL; } }
        PyList_SET_ITEM(lst,i,v);
    }
    return lst;
}

/* ══════════════════════════════════════════════════════════════
/* ══════════════════════════════════════════════════════════════
 *  OMP Parallel Dispatch  (v5.7.6 — high threshold, coarse chunks)
 *  ──────────────────────────────────────────────────────────────
 *  OMP fork/join on Colab Xeon costs ~1-3ms per call. That overhead
 *  only pays off for texts >512KB where parallel work = ~5ms.
 *  Below 512KB, serial is faster. Use max_t chunks (not max_t*4).
 * ══════════════════════════════════════════════════════════════ */
#define PAR_THRESHOLD       (512 * 1024)   /* 512KB: OMP overhead only pays off here */
#define PAR_MIN_THREADS     2
#define MAX_PAR_CHUNKS      MAX_PAR_THREADS

static size_t split_at_ws(const uint8_t *t, size_t target, size_t len) {
    if (target>=len) return len;
    size_t p=target;
    size_t lo=(target>64)?target-64:0;
    while (p>lo) {
        if (t[p]==' '||t[p]=='\n'||t[p]=='\t'||t[p]=='\r') return p+1;
        p--;
    }
    return target;
}

static void tokenize_dispatch(const char *text, size_t len, TBuf *result) {
#ifdef _OPENMP
    if (len >= PAR_THRESHOLD) {
        int max_t = omp_get_max_threads();
        if (max_t > MAX_PAR_THREADS) max_t = MAX_PAR_THREADS;
        if (max_t >= PAR_MIN_THREADS) {
            /* Use exactly max_t chunks — coarse split avoids OMP overhead */
            int nchunks = max_t;
            if (nchunks > MAX_PAR_CHUNKS) nchunks = MAX_PAR_CHUNKS;

            size_t starts[MAX_PAR_CHUNKS + 1];
            starts[0] = 0;
            for (int i = 1; i < nchunks; ++i)
                starts[i] = split_at_ws((const uint8_t*)text,
                                        len * (size_t)i / (size_t)nchunks, len);
            starts[nchunks] = len;

            _ensure_par_caches();

            TBuf chunks[MAX_PAR_THREADS];
            for (int i = 0; i < nchunks; ++i) {
                size_t clen = starts[i+1] - starts[i];
                tb_init(&chunks[i], clen / 2 + 64);
            }

            Py_BEGIN_ALLOW_THREADS
            #pragma omp parallel for num_threads(max_t) schedule(static)
            for (int i = 0; i < nchunks; ++i) {
                int tid = omp_get_thread_num();
                if (tid >= MAX_PAR_THREADS) tid = 0;
                WCEntry *twc = g_par_wc[tid] ? g_par_wc[tid] : g_par_wc[0];
                size_t s = starts[i], e = starts[i+1];
                tokenize_one((const uint8_t*)text + s, e - s, &chunks[i], twc);
            }
            Py_END_ALLOW_THREADS

            /* Pre-sized merge: compute total, ONE realloc, N memcpys */
            size_t total = 0;
            for (int i = 0; i < nchunks; ++i) total += chunks[i].n;
            if (result->cap < result->n + total) {
                result->cap = result->n + total;
                result->d = (int32_t *)realloc(result->d, result->cap * sizeof(int32_t));
            }
            for (int i = 0; i < nchunks; ++i) {
                if (chunks[i].n > 0 && chunks[i].d)
                    memcpy(result->d + result->n, chunks[i].d, chunks[i].n * sizeof(int32_t));
                result->n += chunks[i].n;
                free(chunks[i].d);
            }
            return;
        }
    }
#endif
    WCEntry *wc = _get_tl_wc();
    if (!wc) {
        static WCEntry *fallback_wc = NULL;
        if (!fallback_wc) fallback_wc = (WCEntry *)calloc(WC_TOTAL, sizeof(WCEntry));
        tokenize_one((const uint8_t*)text, len, result, fallback_wc);
        return;
    }
    tokenize_one((const uint8_t*)text, len, result, wc);
}

/* ══════════════════════════════════════════════════════════════
 *  Python API: load_dat(buffer)
 * ══════════════════════════════════════════════════════════════ */
static PyObject *py_load_dat(PyObject *self, PyObject *args) {
    PyObject *buf_obj;
    if (!PyArg_ParseTuple(args,"O",&buf_obj)) return NULL;
    if (g_pybuf_held) { PyBuffer_Release(&g_pybuf); g_pybuf_held=0; }
    Py_XDECREF(g_buf_owner); g_buf_owner=NULL;
    if (PyObject_GetBuffer(buf_obj,&g_pybuf,PyBUF_SIMPLE)!=0) {
        PyErr_SetString(PyExc_TypeError,"Expected buffer"); return NULL;
    }
    g_pybuf_held=1; Py_INCREF(buf_obj); g_buf_owner=buf_obj;
    const char *raw=(const char*)g_pybuf.buf; Py_ssize_t bl=g_pybuf.len;
    if (bl<12||memcmp(raw,"CRAY",4)!=0) {
        PyErr_SetString(PyExc_ValueError,"Invalid DAT magic"); return NULL;
    }
    uint32_t sz=*(const uint32_t*)(raw+8);
    if ((size_t)bl<12+(size_t)sz*12) {
        PyErr_SetString(PyExc_ValueError,"DAT too small"); return NULL;
    }
    const char *arr=raw+12; size_t ab=(size_t)sz*4;
    g_base   =(const int32_t*)(arr);
    g_check  =(const int32_t*)(arr+ab);
    g_values =(const int32_t*)(arr+2*ab);
    g_size   =sz;
    _build_byte_lut();
    _build_pair_lut();
    _build_isword();
    _clear_par_caches();

    int32_t maxv=0;
    for (uint32_t i=0;i<sz;++i) if (g_values[i]>maxv) maxv=g_values[i];
    if (_build_intcache((uint32_t)(maxv+2))!=0) { PyErr_NoMemory(); return NULL; }
    return PyLong_FromUnsignedLong(sz);
}

/* ══════════════════════════════════════════════════════════════
 *  Python API
 * ══════════════════════════════════════════════════════════════ */
static PyObject *py_tokenize(PyObject *self, PyObject *args) {
    const char *text; Py_ssize_t len;
    if (!PyArg_ParseTuple(args,"s#",&text,&len)) return NULL;
    if (!g_size) { PyErr_SetString(PyExc_RuntimeError,"call load_dat() first"); return NULL; }
    TBuf r; tb_init(&r,(size_t)(len/3+64));
    tokenize_dispatch(text,(size_t)len,&r);
    PyObject *res=tb_to_numpy(&r); tb_free(&r); return res;
}

static PyObject *py_tokenize_to_list(PyObject *self, PyObject *args) {
    const char *text; Py_ssize_t len;
    if (!PyArg_ParseTuple(args,"s#",&text,&len)) return NULL;
    if (!g_size) { PyErr_SetString(PyExc_RuntimeError,"call load_dat() first"); return NULL; }
    TBuf r; tb_init(&r,(size_t)(len/3+64));
    tokenize_dispatch(text,(size_t)len,&r);
    PyObject *res=tb_to_pylist(&r); tb_free(&r); return res;
}

static PyObject *py_tokenize_batch(PyObject *self, PyObject *args) {
    PyObject *sl;
    if (!PyArg_ParseTuple(args,"O!",&PyList_Type,&sl)) return NULL;
    if (!g_size) { PyErr_SetString(PyExc_RuntimeError,"call load_dat() first"); return NULL; }
    Py_ssize_t n=PyList_GET_SIZE(sl);
    if (!n) return PyList_New(0);
    const char **txts=(const char**)malloc((size_t)n*sizeof(char*));
    Py_ssize_t  *lens=(Py_ssize_t* )malloc((size_t)n*sizeof(Py_ssize_t));
    if (!txts||!lens) { free(txts);free(lens); return PyErr_NoMemory(); }
    for (Py_ssize_t i=0;i<n;++i) {
        PyObject *it=PyList_GET_ITEM(sl,i);
        if (!PyUnicode_Check(it)) { free(txts);free(lens);
            PyErr_Format(PyExc_TypeError,"Item %zd not str",i); return NULL; }
        txts[i]=PyUnicode_AsUTF8AndSize(it,&lens[i]);
        if (!txts[i]) { free(txts);free(lens); return NULL; }
    }
    TBuf *bufs=(TBuf*)calloc((size_t)n,sizeof(TBuf));
    if (!bufs) { free(txts);free(lens); return PyErr_NoMemory(); }
    _ensure_par_caches();
    Py_BEGIN_ALLOW_THREADS
#ifdef _OPENMP
    #pragma omp parallel
    {
        int tid=omp_get_thread_num();
        if (tid >= MAX_PAR_THREADS) tid = 0;
        WCEntry *wc = g_par_wc[tid] ? g_par_wc[tid] : g_par_wc[0];
        #pragma omp for schedule(dynamic,1)
        for (Py_ssize_t i=0;i<n;++i) {
            tb_init(&bufs[i],(size_t)(lens[i]/3+64));
            tokenize_one((const uint8_t*)txts[i],(size_t)lens[i],&bufs[i],wc);
        }
    }
#else
    {
        WCEntry *wc = _get_tl_wc();
        for (Py_ssize_t i=0;i<n;++i) {
            tb_init(&bufs[i],(size_t)(lens[i]/3+64));
            tokenize_one((const uint8_t*)txts[i],(size_t)lens[i],&bufs[i],wc);
        }
    }
#endif
    Py_END_ALLOW_THREADS
    PyObject *res=PyList_New(n);
    if (!res) { for (Py_ssize_t i=0;i<n;++i) tb_free(&bufs[i]); free(bufs);free(txts);free(lens); return NULL; }
    for (Py_ssize_t i=0;i<n;++i) {
        PyObject *sub=tb_to_numpy(&bufs[i]); tb_free(&bufs[i]);
        if (!sub) { Py_DECREF(res);
            for (Py_ssize_t j=i+1;j<n;++j) tb_free(&bufs[j]);
            free(bufs);free(txts);free(lens); return NULL; }
        PyList_SET_ITEM(res,i,sub);
    }
    free(bufs);free(txts);free(lens); return res;
}

static PyObject *py_tokenize_batch_to_list(PyObject *self, PyObject *args) {
    PyObject *sl;
    if (!PyArg_ParseTuple(args,"O!",&PyList_Type,&sl)) return NULL;
    if (!g_size) { PyErr_SetString(PyExc_RuntimeError,"call load_dat() first"); return NULL; }
    Py_ssize_t n=PyList_GET_SIZE(sl);
    if (!n) return PyList_New(0);
    const char **txts=(const char**)malloc((size_t)n*sizeof(char*));
    Py_ssize_t  *lens=(Py_ssize_t* )malloc((size_t)n*sizeof(Py_ssize_t));
    if (!txts||!lens) { free(txts);free(lens); return PyErr_NoMemory(); }
    for (Py_ssize_t i=0;i<n;++i) {
        PyObject *it=PyList_GET_ITEM(sl,i);
        if (!PyUnicode_Check(it)) { free(txts);free(lens);
            PyErr_Format(PyExc_TypeError,"Item %zd not str",i); return NULL; }
        txts[i]=PyUnicode_AsUTF8AndSize(it,&lens[i]);
        if (!txts[i]) { free(txts);free(lens); return NULL; }
    }
    TBuf *bufs=(TBuf*)calloc((size_t)n,sizeof(TBuf));
    if (!bufs) { free(txts);free(lens); return PyErr_NoMemory(); }
    Py_BEGIN_ALLOW_THREADS
    {
        WCEntry *wc = _get_tl_wc();
        for (Py_ssize_t i=0;i<n;++i) {
            tb_init(&bufs[i],(size_t)(lens[i]/3+64));
            tokenize_one((const uint8_t*)txts[i],(size_t)lens[i],&bufs[i],wc);
        }
    }
    Py_END_ALLOW_THREADS
    PyObject *res=PyList_New(n);
    if (!res) { for (Py_ssize_t i=0;i<n;++i) tb_free(&bufs[i]); free(bufs);free(txts);free(lens); return NULL; }
    for (Py_ssize_t i=0;i<n;++i) {
        PyObject *sub=tb_to_pylist(&bufs[i]); tb_free(&bufs[i]);
        if (!sub) { Py_DECREF(res);
            for (Py_ssize_t j=i+1;j<n;++j) tb_free(&bufs[j]);
            free(bufs);free(txts);free(lens); return NULL; }
        PyList_SET_ITEM(res,i,sub);
    }
    free(bufs);free(txts);free(lens); return res;
}

/* ══════════════════════════════════════════════════════════════
 *  Python API: get_hardware_info
 * ══════════════════════════════════════════════════════════════ */
static PyObject *py_get_hardware_info(PyObject *self, PyObject *args) {
    char brand[49]={0};
#if HAVE_X86
    unsigned int a=0,b=0,c=0,d=0;
    if (__get_cpuid_max(0x80000000u,NULL)>=0x80000004u) {
        __get_cpuid(0x80000002u,&a,&b,&c,&d);
        memcpy(brand,   &a,4);memcpy(brand+ 4,&b,4);memcpy(brand+ 8,&c,4);memcpy(brand+12,&d,4);
        __get_cpuid(0x80000003u,&a,&b,&c,&d);
        memcpy(brand+16,&a,4);memcpy(brand+20,&b,4);memcpy(brand+24,&c,4);memcpy(brand+28,&d,4);
        __get_cpuid(0x80000004u,&a,&b,&c,&d);
        memcpy(brand+32,&a,4);memcpy(brand+36,&b,4);memcpy(brand+40,&c,4);memcpy(brand+44,&d,4);
    }
    size_t e=strlen(brand);
    while (e>0&&brand[e-1]==' ') e--;
    brand[e]='\0';
#endif
    if (!brand[0]) strcpy(brand,"Unknown CPU");
    char info[320];
    snprintf(info,sizeof(info),"%s [Turbo/v5.7.6/L2-Cache+%s/%s 4Ksets×2ways intcache=%u]",
             brand,
#if HAVE_AVX2
             "AVX2",
#else
             "Scalar",
#endif
#ifdef _OPENMP
             "OMP",
#else
             "ST",
#endif
             g_intcache_sz);
    return PyUnicode_FromString(info);
}

/* ══════════════════════════════════════════════════════════════
 *  Module
 * ══════════════════════════════════════════════════════════════ */
static PyMethodDef methods[] = {
    {"tokenize",               py_tokenize,               METH_VARARGS, "→ numpy.ndarray[int32]"},
    {"tokenize_to_list",       py_tokenize_to_list,       METH_VARARGS, "→ list[int] (compat)"},
    {"tokenize_batch",         py_tokenize_batch,         METH_VARARGS, "→ list[ndarray] (OMP)"},
    {"tokenize_batch_to_list", py_tokenize_batch_to_list, METH_VARARGS, "→ list[list[int]]"},
    {"load_dat",               py_load_dat,               METH_VARARGS, "load DAT + build caches"},
    {"get_hardware_info",      py_get_hardware_info,      METH_VARARGS, "hw info string"},
    {NULL,NULL,0,NULL}
};
static struct PyModuleDef moddef = {
    PyModuleDef_HEAD_INIT,"crayon_turbo",
    "CRAYON Turbo v5.7.4: L2-Cache+AVX2+OMP+numpy", -1, methods
};
PyMODINIT_FUNC PyInit_crayon_turbo(void) {
    memset(g_par_wc,0,sizeof(g_par_wc));
    import_array();
    return PyModule_Create(&moddef);
}
