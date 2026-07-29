# Honest Crayon Turbo Benchmark v2.0
# Size sweep: 0.5 KB → 4 MB
# Uses clean vocab.tokenize() | non-repetitive multi-domain corpus | GC enabled | vs tiktoken

import sys
import time
import statistics
import urllib.request
from typing import List, Tuple, Callable

try:
    from crayon import CrayonVocab
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "xerv-crayon"])
    from crayon import CrayonVocab

try:
    import tiktoken
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tiktoken"])
    import tiktoken


def fetch_text(url: str, max_bytes: int = 2_500_000) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read(max_bytes + 1024)
            return data[:max_bytes].decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  Warning: could not fetch {url[:70]}... ({e})")
        return ""


def build_diverse_corpus() -> str:
    print("Building diverse multi-domain corpus...")

    sources = [
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "https://www.gutenberg.org/files/1342/1342-0.txt",
        "https://www.gutenberg.org/files/84/84-0.txt",
        "https://www.gutenberg.org/files/11/11-0.txt",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/asyncio/base_events.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/json/__init__.py",
        "https://raw.githubusercontent.com/torvalds/linux/master/kernel/sched/core.c",
        "https://raw.githubusercontent.com/golang/go/master/src/runtime/proc.go",
        "https://raw.githubusercontent.com/deepmind/dm-haiku/main/README.md",
        "https://raw.githubusercontent.com/pytorch/pytorch/main/README.md",
        "https://raw.githubusercontent.com/numpy/numpy/main/numpy/_core/src/multiarray/multiarraymodule.c",
        "https://raw.githubusercontent.com/python/cpython/main/Python/ceval.c",
    ]

    chunks = []
    for url in sources:
        text = fetch_text(url, max_bytes=600_000)
        if text.strip():
            chunks.append(text)
            print(f"  + {len(text):,} chars ← {url.split('/')[-1]}")

    if not chunks:
        base = (
            "The philosophy of science examines the foundations, methods and implications of science. "
            "Quantum mechanics describes physical properties at atomic and subatomic scales. "
            "Classical physics remains an excellent approximation at ordinary scales.\n"
            "def tokenize(text: str) -> list[int]:\n    return [ord(c) for c in text]\n"
            "fn main() { println!(\"hello from rust\"); }\n"
        )
        chunks = [base * 300]

    corpus = "\n\n".join(chunks)
    print(f"Final unique corpus size: {len(corpus):,} characters\n")
    return corpus


def make_payload(corpus: str, target_bytes: int) -> str:
    if target_bytes <= len(corpus):
        return corpus[:target_bytes]
    reps = (target_bytes // len(corpus)) + 1
    return (corpus * reps)[:target_bytes]


def time_one_payload(
    fn: Callable,
    payload: str,
    iterations: int,
) -> Tuple[float, float, List[float], int]:
    # Honest warm-up
    tokens = fn(payload)
    token_count = len(tokens)

    latencies = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        tokens = fn(payload)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    mean_latency_s = statistics.mean(latencies) / 1000
    mean_tps = token_count / mean_latency_s
    mean_mbps = (len(payload.encode("utf-8")) / (1024 * 1024)) / mean_latency_s
    return mean_tps, mean_mbps, latencies, token_count


def print_row(size_name: str, name: str, tps: float, mbps: float,
              latencies: List[float], tokens: int):
    mean_lat = statistics.mean(latencies)
    std_lat = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    print(
        f"| {size_name:<10} | {name:<22} | {tokens:>10,} | "
        f"{tps/1e6:>8.2f} M | {mbps:>7.1f} | "
        f"{mean_lat:>7.2f} ± {std_lat:>5.2f} |"
    )


def main():
    print("=" * 110)
    print("HONEST CRAYON TURBO BENCHMARK  –  Size Sweep (0.5 KB → 4 MB)")
    print("Uses vocab.tokenize()  |  minimal repetition  |  GC enabled  |  vs tiktoken")
    print("=" * 110)

    # Official Crayon API
    vocab = CrayonVocab(device="auto")
    vocab.load_profile("lite")
    
    print(f"Device Mode   : {vocab.device}")
    print(f"Turbo Engine  : {vocab._turbo_backend.get_hardware_info() if vocab._turbo_backend else 'N/A'}\n")

    corpus = build_diverse_corpus()

    sizes = [
        ("0.5 KB",   512),
        ("1 KB",    1024),
        ("2 KB",    2048),
        ("4 KB",    4096),
        ("8 KB",    8192),
        ("16 KB",  16384),
        ("32 KB",  32768),
        ("64 KB",  65536),
        ("128 KB", 131072),
        ("256 KB", 262144),
        ("512 KB", 524288),
        ("1 MB",  1048576),
        ("2 MB",  2097152),
        ("4 MB",  4194304),
    ]

    enc = tiktoken.get_encoding("cl100k_base")

    def crayon_fn(text: str):
        return vocab.tokenize(text)

    def tiktoken_fn(text: str):
        return enc.encode(text)

    print("-" * 110)
    print(
        f"| {'Size':<10} | {'Tokenizer':<22} | {'Tokens':>10} | "
        f"{'Throughput':>10} | {'MB/s':>7} | {'Latency (ms)':^15} |"
    )
    print(
        f"| {'':<10} | {'':<22} | {'':>10} | "
        f"{'(M tok/s)':>10} | {'':>7} | {'mean ± std':^15} |"
    )
    print("-" * 110)

    for size_name, num_bytes in sizes:
        payload = make_payload(corpus, num_bytes)

        iterations = 40 if num_bytes < 32_768 else 12 if num_bytes < 524_288 else 7

        tps, mbps, lats, ntok = time_one_payload(crayon_fn, payload, iterations)
        print_row(size_name, "Crayon Turbo", tps, mbps, lats, ntok)

        tps, mbps, lats, ntok = time_one_payload(tiktoken_fn, payload, iterations)
        print_row(size_name, "tiktoken cl100k", tps, mbps, lats, ntok)

        print("-" * 110)

    print("\nMethodology notes (honest by design):")
    print("  • Diverse multi-domain corpus (literature + real C/Python/Go source + technical docs)")
    print("  • Minimal repetition: only repeats the unique corpus when target size > corpus length")
    print("  • Garbage collector left fully enabled")
    print("  • Warm-up = one pass only")
    print("  • More iterations on small sizes for stable statistics")
    print("  • Same payload fed to both tokenizers")
    print("  • Official Crayon API: vocab.tokenize(text)")
    print("=" * 110)


if __name__ == "__main__":
    main()
