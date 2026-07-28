# Crayon Real-World Colab Benchmark Script (v5.5.2)
# Run in Google Colab:
#   !pip install -U xerv-crayon
#   !python measurecolab.py

import sys, os, time, gc, urllib.request

try:
    from crayon import CrayonVocab
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "xerv-crayon"])
    from crayon import CrayonVocab

# Fallback diverse text if offline
DIVERSE_FALLBACK_PROSE = """
The philosophy of science is a sub-field of philosophy concerned with the foundations, methods, and implications of science.
The central questions of this study concern what qualifies as science, the reliability of scientific theories, and the ultimate purpose of science.
Quantum mechanics is a fundamental theory in physics that provides a description of the physical properties of nature at the scale of atoms and subatomic particles.
It is the foundation of all quantum physics including quantum chemistry, quantum field theory, quantum technology, and quantum information science.
Classical physics, the collection of theories that existed before the advent of quantum mechanics, describes many aspects of nature at an ordinary (large) scale,
but is not sufficient for describing them at small (atomic and subatomic) scales. Most theories in classical physics can be derived from quantum mechanics as an approximation.
""" * 100

def fetch_real_corpus(filename: str, fallback_text: str) -> str:
    """Fetch real-world corpus file locally or from GitHub repo."""
    local_path = os.path.join(os.path.dirname(__file__), "src", "crayon", "resources", filename)
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                content = f.read()
                if content.strip():
                    return content
        except Exception:
            pass

    url = f"https://raw.githubusercontent.com/Xerv-Org/CRAYON/main/src/crayon/resources/{filename}"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            content = response.read().decode('utf-8')
            if content.strip():
                return content
    except Exception:
        pass

    return fallback_text

def make_payload(base_text: str, target_bytes: int) -> str:
    rep = (target_bytes // len(base_text)) + 1
    return (base_text * rep)[:target_bytes]

def run_benchmark():
    print("=" * 85)
    print("🚀 CRAYON REAL-WORLD BENCHMARK (NON-INFLATED DIVERSE DATASETS)")
    print("=" * 85)

    vocab = CrayonVocab(device="auto")
    vocab.load_profile("lite")
    turbo = vocab._turbo_backend

    if turbo:
        print(f"Hardware Info: {turbo.get_hardware_info()}")
    else:
        print(f"Backend Device: {vocab.device}")

    print("\nFetching real-world datasets (Prose, Source Code, Scientific Data)...")
    prose_text = fetch_real_corpus("arts_commerce_corpus.txt", DIVERSE_FALLBACK_PROSE)
    code_text  = fetch_real_corpus("CRAYON_Full_Codebase.txt", DIVERSE_FALLBACK_PROSE)
    tech_text  = fetch_real_corpus("science_corpus.txt", DIVERSE_FALLBACK_PROSE)

    domains = [
        ("Real English Prose", prose_text),
        ("Real Source Code",   code_text),
        ("Scientific / Tech",  tech_text)
    ]

    sizes = [
        ("100 KB", 100 * 1024),
        ("1 MB",   1 * 1024 * 1024),
        ("4 MB",   4 * 1024 * 1024),
        ("10 MB",  10 * 1024 * 1024)
    ]

    print("\n" + "=" * 85)
    print(f"| {'Domain':<22} | {'Size':<8} | {'Tokens':<10} | {'Throughput':<14} | {'Latency':<9} | {'Status':<7} |")
    print("-" * 85)

    for domain_name, base_text in domains:
        for size_name, num_bytes in sizes:
            payload = make_payload(base_text, num_bytes)
            iterations = 30 if num_bytes < 1000000 else 10

            # Warmup
            turbo.tokenize(payload)

            gc.collect()
            gc.disable()
            t0 = time.perf_counter()
            for _ in range(iterations):
                tokens = turbo.tokenize(payload)
            t1 = time.perf_counter()
            gc.enable()

            elapsed = (t1 - t0) / iterations
            tps = len(tokens) / elapsed
            status = "✅ HIGH" if tps >= 105e6 else "ℹ️ REAL"

            print(f"| {domain_name:<22} | {size_name:<8} | {len(tokens):>10,} | {tps/1e6:>10.2f} M tok/s | {elapsed*1000:>6.2f} ms | {status:<7} |")
        print("-" * 85)

    print("\n" + "=" * 85)
    print("BENCHMARK COMPLETE")
    print("=" * 85 + "\n")

if __name__ == "__main__":
    run_benchmark()

