import re

with open('/home/soham/CRAYON/src/crayon/c_ext/crayon_turbo.c', 'r') as f:
    code = f.read()

# Change WORD_CACHE_SIZE to 64K
code = re.sub(r'#define WORD_CACHE_SIZE \(1u << 18\)', '#define WORD_CACHE_SIZE (1u << 16)', code)

# Change PAR_THRESHOLD to 256KB
code = re.sub(r'#define PAR_THRESHOLD \(32 \* 1024\)', '#define PAR_THRESHOLD (256 * 1024)', code)

# Change TARGET_MICRO_CHUNK to 256KB
code = re.sub(r'#define TARGET_MICRO_CHUNK \(32 \* 1024\)', '#define TARGET_MICRO_CHUNK (256 * 1024)', code)

with open('/home/soham/CRAYON/src/crayon/c_ext/crayon_turbo.c', 'w') as f:
    f.write(code)

