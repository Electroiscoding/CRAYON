import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _mb_per_sec(byte_count: int, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return (byte_count / 1024.0 / 1024.0) / seconds


@dataclass
class BenchCase:
    name: str
    text: str
    repeat: int = 1


@dataclass
class BenchResult:
    impl: str
    case: str
    status: str
    load_time_ms: float
    tokens_produced: int
    bytes_processed: int
    avg_time_ms: float
    tokens_per_sec: float
    mb_per_sec: float
    notes: str = ""


def _default_cases() -> List[BenchCase]:
    english = (
        "The quick brown fox jumps over the lazy dog. "
        "Tokenization benchmarks should include punctuation, numbers 12345, and whitespace. "
        "This is a medium length sentence for throughput testing. "
    )
    code = (
        "def matrix_multiply(A, B):\n"
        "    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]\n"
        "    for i in range(len(A)):\n"
        "        for j in range(len(B[0])):\n"
        "            for k in range(len(B)):\n"
        "                result[i][j] += A[i][k] * B[k][j]\n"
        "    return result\n"
    )
    unicode = (
        "E=mc². हिंदी: द. عربى: مرحبا. 中文: 你好. emoji: 😀🚀✨. "
        "Combining marks: a."
    )
    mixed = english + "\n" + code + "\n" + unicode

    return [
        BenchCase(name="english", text=english, repeat=4000),
        BenchCase(name="code", text=code, repeat=4000),
        BenchCase(name="unicode", text=unicode, repeat=6000),
        BenchCase(name="mixed", text=mixed, repeat=2500),
    ]


def _run_single(
    *,
    impl_name: str,
    case: BenchCase,
    load_fn: Callable[[], Any],
    tokenize_fn: Callable[[str], Sequence[int]],
    iterations: int,
    warmup: int,
) -> BenchResult:
    try:
        t0 = time.perf_counter()
        load_fn()
        load_ms = (time.perf_counter() - t0) * 1000.0

        payload = case.text * case.repeat
        payload_bytes = payload.encode("utf-8")

        for _ in range(warmup):
            _ = tokenize_fn(payload)

        total_t = 0.0
        total_tokens = 0
        for _ in range(iterations):
            s = time.perf_counter()
            toks = tokenize_fn(payload)
            total_t += (time.perf_counter() - s)
            total_tokens += len(toks)

        avg_t = total_t / max(iterations, 1)
        avg_tokens = int(total_tokens / max(iterations, 1))

        tps = (avg_tokens / avg_t) if avg_t > 0 else 0.0
        mbs = _mb_per_sec(len(payload_bytes), avg_t)

        return BenchResult(
            impl=impl_name,
            case=case.name,
            status="OK",
            load_time_ms=load_ms,
            tokens_produced=avg_tokens,
            bytes_processed=len(payload_bytes),
            avg_time_ms=avg_t * 1000.0,
            tokens_per_sec=tps,
            mb_per_sec=mbs,
        )
    except Exception as e:
        return BenchResult(
            impl=impl_name,
            case=case.name,
            status="FAIL",
            load_time_ms=0.0,
            tokens_produced=0,
            bytes_processed=0,
            avg_time_ms=0.0,
            tokens_per_sec=0.0,
            mb_per_sec=0.0,
            notes=str(e),
        )


def _try_crayon_impl(device: str, profile: str) -> Optional[Tuple[str, Callable[[], Any], Callable[[str], Sequence[int]]]]:
    try:
        sys.path.insert(0, os.path.join(os.getcwd(), "src"))
        from crayon.core.vocabulary import CrayonVocab
    except Exception:
        return None

    name = f"crayon:{device}:{profile}"
    vocab: Optional[Any] = None

    def load() -> Any:
        nonlocal vocab
        vocab = CrayonVocab(device=device)
        vocab.load_profile(profile)
        return vocab

    def tokenize(text: str) -> Sequence[int]:
        if vocab is None:
            raise RuntimeError("CrayonVocab not loaded")
        return vocab.tokenize(text)  # type: ignore[return-value]

    return name, load, tokenize


def _try_tiktoken_impl(encoding_name: str) -> Optional[Tuple[str, Callable[[], Any], Callable[[str], Sequence[int]]]]:
    try:
        import tiktoken
    except Exception:
        return None

    name = f"tiktoken:{encoding_name}"
    enc: Optional[Any] = None

    def load() -> Any:
        nonlocal enc
        enc = tiktoken.get_encoding(encoding_name)
        return enc

    def tokenize(text: str) -> Sequence[int]:
        if enc is None:
            raise RuntimeError("tiktoken encoding not loaded")
        return enc.encode(text)

    return name, load, tokenize


def _try_hf_impl(model_id: str) -> Optional[Tuple[str, Callable[[], Any], Callable[[str], Sequence[int]]]]:
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None

    name = f"hf:{model_id}"
    tok: Optional[Any] = None

    def load() -> Any:
        nonlocal tok
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        return tok

    def tokenize(text: str) -> Sequence[int]:
        if tok is None:
            raise RuntimeError("HF tokenizer not loaded")
        return tok.encode(text)

    return name, load, tokenize


def _write_outputs(results: List[BenchResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in results], f, ensure_ascii=False, indent=2)

    csv_path = out_dir / "benchmark_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(BenchResult.__dataclass_fields__.keys()))
        w.writeheader()
        for r in results:
            w.writerow(r.__dict__)


def _plot(results: List[BenchResult], out_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    ok = [r for r in results if r.status == "OK"]
    if not ok:
        return

    impls = sorted(set(r.impl for r in ok))
    cases = sorted(set(r.case for r in ok))

    def metric_matrix(metric: str) -> List[List[float]]:
        m: List[List[float]] = []
        for c in cases:
            row: List[float] = []
            for i in impls:
                v = next((getattr(r, metric) for r in ok if r.impl == i and r.case == c), 0.0)
                row.append(float(v))
            m.append(row)
        return m

    def bar_by_case(metric: str, title: str, fname: str) -> None:
        width = 0.8 / max(len(impls), 1)
        x = list(range(len(cases)))

        fig = plt.figure(figsize=(max(10, len(cases) * 2), 6))
        ax = fig.add_subplot(111)

        for idx, impl in enumerate(impls):
            vals = [
                next((float(getattr(r, metric)) for r in ok if r.impl == impl and r.case == c), 0.0)
                for c in cases
            ]
            ax.bar([xi + idx * width for xi in x], vals, width=width, label=impl)

        ax.set_title(title)
        ax.set_xticks([xi + (len(impls) * width) / 2 for xi in x])
        ax.set_xticklabels(cases, rotation=15, ha="right")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=200)
        plt.close(fig)

    bar_by_case("tokens_per_sec", "Tokens/sec (higher is better)", "tokens_per_sec.png")
    bar_by_case("mb_per_sec", "MB/sec (higher is better)", "mb_per_sec.png")
    bar_by_case("load_time_ms", "Load time (ms) (lower is better)", "load_time_ms.png")
    bar_by_case("tokens_produced", "Tokens produced (avg per run)", "tokens_produced.png")


def main() -> int:
    ap = argparse.ArgumentParser(prog="benchmark_suite")
    ap.add_argument("--device", default="cpu", choices=["cpu", "auto", "cuda", "rocm"])
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--out", default=str(Path("benchmark_results") / _now_tag()))
    ap.add_argument("--include-hf", action="store_true")
    args = ap.parse_args()

    cases = _default_cases()

    impls: List[Tuple[str, Callable[[], Any], Callable[[str], Sequence[int]]]] = []

    for profile in ["lite", "standard"]:
        cr = _try_crayon_impl(args.device, profile)
        if cr is not None:
            impls.append(cr)

    for enc_name in ["p50k_base", "cl100k_base", "o200k_base"]:
        tk = _try_tiktoken_impl(enc_name)
        if tk is not None:
            impls.append(tk)

    if args.include_hf:
        for model_id in [
            "gpt2",
            "bert-base-uncased",
        ]:
            hf = _try_hf_impl(model_id)
            if hf is not None:
                impls.append(hf)

    results: List[BenchResult] = []

    print("=" * 90)
    print("CRAYON BENCHMARK SUITE")
    print("=" * 90)
    print(f"Device: {args.device}")
    print(f"Iterations: {args.iterations} | Warmup: {args.warmup}")
    print(f"Output: {args.out}")
    print("Implementations:")
    for n, _, _ in impls:
        print(f"  - {n}")
    print("Cases:")
    for c in cases:
        approx_mb = len((c.text * c.repeat).encode("utf-8")) / 1024.0 / 1024.0
        print(f"  - {c.name}: ~{approx_mb:.2f} MB")
    print("-" * 90)

    for impl_name, load_fn, tok_fn in impls:
        for case in cases:
            r = _run_single(
                impl_name=impl_name,
                case=case,
                load_fn=load_fn,
                tokenize_fn=tok_fn,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            results.append(r)
            if r.status == "OK":
                print(
                    f"[OK] {r.impl:<22} {r.case:<8} "
                    f"load={r.load_time_ms:>8.2f}ms "
                    f"avg={r.avg_time_ms:>8.2f}ms "
                    f"tok={r.tokens_produced:>8} "
                    f"tps={r.tokens_per_sec:>12.0f} "
                    f"mbps={r.mb_per_sec:>8.2f}"
                )
            else:
                print(f"[FAIL] {r.impl:<22} {r.case:<8} {r.notes}")

    out_dir = Path(args.out)
    _write_outputs(results, out_dir)
    _plot(results, out_dir)

    print("-" * 90)
    print("WROTE:")
    print(f"  - {out_dir / 'benchmark_results.json'}")
    print(f"  - {out_dir / 'benchmark_results.csv'}")
    print(f"  - {out_dir / 'tokens_per_sec.png'} (if matplotlib installed)")
    print(f"  - {out_dir / 'mb_per_sec.png'} (if matplotlib installed)")
    print(f"  - {out_dir / 'load_time_ms.png'} (if matplotlib installed)")
    print(f"  - {out_dir / 'tokens_produced.png'} (if matplotlib installed)")
    print("=" * 90)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
