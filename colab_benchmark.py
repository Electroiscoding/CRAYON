import importlib
import subprocess
import sys
import time


PKG = "xerv-crayon==4.1.9"


def ensure_crayon():
    try:
        import crayon

        if getattr(crayon, "__version__", None) == PKG.split("==", 1)[1]:
            return crayon
    except Exception:
        pass
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            PKG,
            "--no-binary",
            ":all:",
            "--upgrade",
            "--no-cache-dir",
            "-v",
        ]
    )
    importlib.invalidate_caches()
    import crayon

    return crayon


def _unwrap_tokens(ret):
    return ret[0] if isinstance(ret, tuple) else ret


def bench(device, batch, total_bytes):
    from crayon import CrayonVocab

    v = CrayonVocab(device=device)
    v.load_profile("lite")
    try:
        v.tokenize(["warmup"])
    except Exception:
        pass
    t0 = time.perf_counter()
    toks = _unwrap_tokens(v.tokenize(batch))
    dt = time.perf_counter() - t0
    total_tokens = sum(len(x) for x in toks)
    mbps = (total_bytes / (1024 * 1024)) / dt if dt > 0 else float("inf")
    tokps = total_tokens / dt if dt > 0 else float("inf")
    return dt, mbps, tokps


def main():
    crayon = ensure_crayon()
    from crayon import CrayonVocab

    target_bytes = 100 * 1024 * 1024
    batch_n = 256
    base = "The quick brown fox jumps over the lazy dog. Programming in Rust and CUDA is fun! "
    per_doc = max(1, target_bytes // batch_n)
    reps = max(1, per_doc // len(base))
    doc = (base * reps)[:per_doc]
    batch = [doc] * batch_n
    total_bytes = sum(len(s) for s in batch)

    auto_dev = None
    try:
        auto_dev = CrayonVocab(device="auto").get_info().get("device")
    except Exception:
        pass
    print(
        f"crayon={getattr(crayon,'__version__',None)} payload_mb={total_bytes/1024/1024:.1f} auto={auto_dev}"
    )

    dt_cpu, mb_cpu, tok_cpu = bench("cpu", batch, total_bytes)
    print(f"cpu_s={dt_cpu:.4f} cpu_MBps={mb_cpu:.1f} cpu_tokps={tok_cpu:.0f}")

    try:
        dt_gpu, mb_gpu, tok_gpu = bench("cuda", batch, total_bytes)
        print(
            f"cuda_s={dt_gpu:.4f} cuda_MBps={mb_gpu:.1f} cuda_tokps={tok_gpu:.0f} speedup={dt_cpu/dt_gpu:.2f}"
        )
    except Exception as e:
        print(f"cuda=skip err={type(e).__name__}:{e}")


if __name__ == "__main__":
    main()
