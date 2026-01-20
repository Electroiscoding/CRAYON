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

# Suppress verbose logging
logging.disable(logging.WARNING)

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), "build", "lib.win-amd64-cpython-313"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from crayon.c_ext.dat_builder import DATBuilder

# Storage locations
PACKAGE_DAT_DIR = Path("src/crayon/resources/dat")
USER_CACHE_DIR = Path.home() / ".cache" / "xerv" / "crayon" / "profiles"

# Vocabulary profiles to build
VOCAB_PROFILES = [
    {
        "name": "science",
        "source": "trained_vocab_science.json",
        "description": "High-Precision Math, Physics & LaTeX Support"
    },
    {
        "name": "code",
        "source": "trained_vocab_code.json",
        "description": "Python, Rust, C++, JavaScript Syntax"
    },
    {
        "name": "multilingual",
        "source": "trained_vocab_multilingual.json",
        "description": "European Languages, Chinese, Hindi"
    },
    {
        "name": "arts_commerce",
        "source": "trained_vocab_arts_commerce.json",
        "description": "Legal, Financial, Literature"
    },
    {
        "name": "lite",
        "source": "trained_vocab_lite.json",
        "description": "General English, 50k tokens, Speed-optimized"
    },
]

def load_vocab(source_path: str) -> list:
    """Load vocabulary from JSON file."""
    with open(source_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [k for k, v in sorted(data.items(), key=lambda x: x[1])]
    else:
        raise ValueError(f"Unknown vocab format in {source_path}")

def build_profile(profile: dict, output_dirs: list) -> dict:
    """Build a single profile and save to all output directories."""
    name = profile["name"]
    source = profile["source"]
    
    if not os.path.exists(source):
        return {"name": name, "status": "SKIP", "reason": f"Source not found: {source}"}
    
    try:
        # Load vocabulary
        vocab = load_vocab(source)
        vocab_size = len(vocab)
        
        # Build DAT
        builder = DATBuilder()
        start = time.perf_counter()
        builder.build(vocab)
        build_time = time.perf_counter() - start
        
        # Save to all output directories
        saved_paths = []
        for output_dir in output_dirs:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save DAT file
            dat_path = output_dir / f"vocab_{name}.dat"
            builder.save(str(dat_path))
            saved_paths.append(str(dat_path))
            
            # Also save JSON for decode() support
            json_path = output_dir / f"vocab_{name}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(vocab, f, ensure_ascii=False)
        
        return {
            "name": name,
            "status": "OK",
            "vocab_size": vocab_size,
            "dat_nodes": builder.size,
            "dat_size_kb": os.path.getsize(saved_paths[0]) / 1024,
            "build_time_s": build_time,
            "paths": saved_paths
        }
        
    except Exception as e:
        return {"name": name, "status": "FAIL", "reason": str(e)}

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
    
    for profile in VOCAB_PROFILES:
        name = profile["name"]
        print(f"[BUILD] {name:<20} ({profile['description'][:40]})", end=" ", flush=True)
        
        result = build_profile(profile, output_dirs)
        results.append(result)
        
        if result["status"] == "OK":
            print(f"✓ {result['vocab_size']:,} tokens → {result['dat_nodes']:,} nodes | {result['build_time_s']:.1f}s")
        elif result["status"] == "SKIP":
            print(f"⊘ SKIPPED: {result['reason']}")
        else:
            print(f"✗ FAILED: {result['reason']}")
    
    print("-" * 80)
    print()
    
    # Summary
    ok_count = sum(1 for r in results if r["status"] == "OK")
    print(f"✅ Successfully built: {ok_count}/{len(VOCAB_PROFILES)} profiles")
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
    print("   1. Commit src/crayon/resources/dat/*.dat to git")
    print("   2. Users can now use: CrayonVocab.load_profile('code')")
    print()

if __name__ == "__main__":
    main()
