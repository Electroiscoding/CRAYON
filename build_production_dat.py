"""
XERV CRAYON V2.0 - Production DAT Builder
Compiles all vocabulary profiles to production-ready .dat files.

Storage Locations:
1. src/crayon/resources/dat/ - For package distribution (checked into git)
2. ~/.cache/xerv/crayon/profiles/ - User cache for runtime

Run this once during development, commit the .dat files to git.
"""
import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, List

# Suppress verbose logging
logging.disable(logging.WARNING)

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), "build", "lib.win-amd64-cpython-313"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Storage locations
PACKAGE_DAT_DIR = Path("src/crayon/resources/dat")
USER_CACHE_DIR = Path.home() / ".cache" / "xerv" / "crayon" / "profiles"


def _tiktoken_vocab(encoding_name: str, limit: int) -> List[str]:
    import tiktoken

    enc = tiktoken.get_encoding(encoding_name)
    n_vocab = int(getattr(enc, "n_vocab", 0))
    if n_vocab <= 0:
        raise RuntimeError(f"tiktoken encoding {encoding_name!r} has invalid n_vocab={n_vocab}")

    out: List[str] = []
    for i in range(n_vocab):
        if len(out) >= limit:
            break
        try:
            out.append(enc.decode([i]))
        except Exception:
            # Some encodings contain special/un-decodable token IDs.
            # We skip them and continue until we hit the requested limit.
            continue

    if len(out) != limit:
        raise RuntimeError(
            f"Failed to collect {limit} decodable tokens from {encoding_name!r}. "
            f"Got {len(out)} (n_vocab={n_vocab})."
        )
    return out


def _build_lite_vocab() -> List[str]:
    return _tiktoken_vocab("p50k_base", 50000)


def _build_standard_vocab() -> List[str]:
    lite = _build_lite_vocab()
    existing = set(lite)

    # Try to add up to 200k tokens from o200k_base, skipping undecodable and duplicates.
    # If that results in <250k total, that's allowed by user request.
    extra = _tiktoken_vocab("o200k_base", 200000)
    merged: List[str] = list(lite)
    for tok in extra:
        if tok in existing:
            continue
        merged.append(tok)
        existing.add(tok)
        if len(merged) >= 250000:
            break

    return merged


def _compile_dat(vocab: List[str], dat_path: Path) -> Dict:
    try:
        from crayon.c_ext import crayon_compiler
    except Exception as e:
        raise RuntimeError(
            "C/C++ DAT compiler extension 'crayon_compiler' is required for build_production_dat.py. "
            "Build/install the package with extensions enabled, then re-run. "
            f"Original error: {e}"
        )

    return crayon_compiler.compile_dat(vocab, str(dat_path))


def build_profile(name: str, vocab: List[str], output_dirs: List[Path]) -> Dict:
    start = time.perf_counter()

    saved_paths = []
    compile_stats: Dict = {}
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"vocab_{name}.json"
        dat_path = output_dir / f"vocab_{name}.dat"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)

        compile_stats = _compile_dat(vocab, dat_path)
        saved_paths.append(str(dat_path))

    build_time = time.perf_counter() - start
    dat_size_kb = os.path.getsize(saved_paths[0]) / 1024
    return {
        "name": name,
        "status": "OK",
        "vocab_size": len(vocab),
        "dat_size_kb": dat_size_kb,
        "build_time_s": build_time,
        "compile_stats": compile_stats,
        "paths": saved_paths,
    }

def main():
    print("=" * 80)
    print("XERV CRAYON V2.0 - PRODUCTION DAT BUILDER")
    print("=" * 80)
    print()
    
    # Output directories
    output_dirs = [PACKAGE_DAT_DIR, USER_CACHE_DIR]
    
    print("📁 Output Locations:")
    for d in output_dirs:
        print(f"   • {d}")
    print()
    
    print("-" * 80)
    results = []
    
    profiles = [
        ("lite", _build_lite_vocab),
        ("standard", _build_standard_vocab),
    ]

    for name, fn in profiles:
        print(f"[BUILD] {name:<20}", end=" ", flush=True)
        try:
            vocab = fn()
            result = build_profile(name, vocab, output_dirs)
            results.append(result)
            print(
                f"✓ {result['vocab_size']:,} tokens | {result['dat_size_kb']:.1f} KB | {result['build_time_s']:.1f}s"
            )
        except Exception as e:
            results.append({"name": name, "status": "FAIL", "reason": str(e)})
            print(f"✗ FAILED: {e}")
    
    print("-" * 80)
    print()
    
    # Summary
    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"✅ Successfully built: {ok_count}/{len(results)} profiles")
    print()
    
    # Show what was created
    print("📦 Files Created:")
    for result in results:
        if result["status"] == "OK":
            print(f"   {result['name']:<20} {result['dat_size_kb']:.1f} KB")
            for path in result["paths"]:
                print(f"      └─ {path}")
    
    print()
    print("=" * 80)
    print("PRODUCTION DAT BUILD COMPLETE")
    print("=" * 80)
    print()
    print("📌 Next Steps:")
    print("   1. Commit src/crayon/resources/dat/vocab_lite.* and vocab_standard.* to git")
    print("   2. Users can now use: CrayonVocab.load_profile('lite'|'standard')")
    print()

if __name__ == "__main__":
    main()
