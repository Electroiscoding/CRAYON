#!/usr/bin/env python3
"""
CRAYON Hyper-Fast Code Profile Trainer
=======================================

Uses the C++ BPE trainer (Linked-List + Inverted Index + Lazy Heap)
to train on the CRAYON codebase with maximum performance.

This is the fastest possible way to train a BPE vocabulary on a single CPU core.

Algorithm Complexity:
- Initial Counting: O(N) where N = corpus size
- Per Merge: O(K * log H) where K = pair frequency, H = heap size
- Total: O(N + M * K_avg * log H) where M = vocab_size - 256

This script:
1. Shows current "code" profile vocabulary stats
2. Trains a new vocabulary using the hyper-fast C++ engine
3. Compares before/after vocabularies
4. Builds DAT file for runtime use
"""

import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from crayon.c_ext.dat_builder import DATBuilder

# Paths
REPO_ROOT = Path(__file__).parent
CODEBASE_FILE = REPO_ROOT / "src" / "crayon" / "resources" / "CRAYON_Full_Codebase.txt"
RESOURCES_DIR = REPO_ROOT / "src" / "crayon" / "resources" / "dat"

# Current code profile paths
CURRENT_JSON = RESOURCES_DIR / "vocab_code.json"
CURRENT_DAT = RESOURCES_DIR / "vocab_code.dat"

# New profile paths (will overwrite)
NEW_JSON = RESOURCES_DIR / "vocab_code.json"
NEW_DAT = RESOURCES_DIR / "vocab_code.dat"


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.2f} ms"
    elif seconds < 60:
        return f"{seconds:.2f} s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def format_size(bytes_count: int) -> str:
    """Format bytes into human-readable string."""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.2f} KB"
    else:
        return f"{bytes_count / (1024 * 1024):.2f} MB"


def load_vocab(path: Path) -> list:
    """Load vocabulary from JSON file."""
    if not path.exists():
        return []
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Handle both list and dict formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # If it's our structured format with 'vocab' key
            if 'vocab' in data:
                return list(data['vocab'].keys())
            # Otherwise treat as token:id dict
            return list(data.keys())
        return []


def analyze_vocab(vocab: list, name: str):
    """Analyze and print vocabulary statistics."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    
    if not vocab:
        print("  (Empty vocabulary)")
        return {}
    
    # Basic stats
    print(f"  Total tokens:     {len(vocab):,}")
    
    # Categorize tokens
    special = [t for t in vocab if t.startswith('<') and t.endswith('>')]
    single_char = [t for t in vocab if len(t) == 1 and t not in special]
    multi_char = [t for t in vocab if len(t) > 1 and t not in special]
    
    # Programming keywords
    py_keywords = {'def', 'class', 'import', 'from', 'return', 'if', 'else', 'elif', 
                   'for', 'while', 'try', 'except', 'with', 'as', 'self', 'None',
                   'True', 'False', 'lambda', 'yield', 'async', 'await', 'pass'}
    cpp_keywords = {'int', 'void', 'char', 'float', 'double', 'bool', 'struct',
                    'class', 'public', 'private', 'return', 'if', 'else', 'for',
                    'while', 'const', 'static', 'inline', 'namespace', 'include'}
    cuda_keywords = {'__global__', '__device__', '__shared__', 'threadIdx', 
                     'blockIdx', 'blockDim', 'gridDim', 'cudaMalloc', 'cudaMemcpy'}
    
    found_py = [t for t in vocab if t in py_keywords]
    found_cpp = [t for t in vocab if t in cpp_keywords]
    found_cuda = [t for t in vocab if t in cuda_keywords]
    
    print(f"  Special tokens:   {len(special)}")
    print(f"  Single chars:     {len(single_char)}")
    print(f"  Multi-char:       {len(multi_char)}")
    print(f"\n  Programming keywords found:")
    print(f"    Python:  {len(found_py)} ({', '.join(found_py[:5])}{'...' if len(found_py) > 5 else ''})")
    print(f"    C/C++:   {len(found_cpp)} ({', '.join(found_cpp[:5])}{'...' if len(found_cpp) > 5 else ''})")
    print(f"    CUDA:    {len(found_cuda)} ({', '.join(found_cuda[:3])}{'...' if len(found_cuda) > 3 else ''})")
    
    # Token length distribution
    lengths = [len(t) for t in vocab]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0
    
    print(f"\n  Token lengths:")
    print(f"    Average: {avg_len:.2f} chars")
    print(f"    Max:     {max_len} chars")
    
    # Sample tokens
    print(f"\n  Sample multi-char tokens (first 20):")
    sample = [t for t in multi_char if len(t) > 2][:20]
    for i in range(0, len(sample), 5):
        row = sample[i:i+5]
        print(f"    {', '.join(repr(t) for t in row)}")
    
    return {
        'total': len(vocab),
        'special': len(special),
        'single': len(single_char),
        'multi': len(multi_char),
        'py_keywords': len(found_py),
        'cpp_keywords': len(found_cpp),
        'cuda_keywords': len(found_cuda),
        'avg_len': avg_len
    }


def train_with_cpp_engine(corpus_bytes: bytes, vocab_size: int, min_freq: int = 2) -> tuple:
    """
    Train vocabulary using the hyper-fast C++ BPE trainer.
    
    Returns:
        Tuple of (vocab_list, merges_list, stats_dict)
    """
    from crayon.c_ext import crayon_trainer
    
    print(f"\n  Trainer Version: {crayon_trainer.get_version()}")
    print(f"  Algorithm: {crayon_trainer.get_algorithm_info().split(chr(10))[0]}")
    
    # Execute training with verbose stats
    start_train = time.perf_counter()
    merges = crayon_trainer.train_fast(
        corpus_bytes,
        vocab_size,
        min_freq=min_freq,
        verbose=1
    )
    train_time = time.perf_counter() - start_train
    
    # Build vocabulary from merges
    # Start with byte tokens (0-255) - printable characters
    vocab = {}
    token_strings = {}
    
    # Add printable ASCII
    for i in range(256):
        try:
            char = chr(i)
            if char.isprintable() and char.strip():
                vocab[char] = i
                token_strings[i] = char
            else:
                # Use hex representation for non-printable
                hex_repr = f"<0x{i:02X}>"
                token_strings[i] = hex_repr
        except (ValueError, UnicodeError):
            token_strings[i] = f"<0x{i:02X}>"
    
    # Add special tokens
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    next_special_id = 256
    for tok in special_tokens:
        vocab[tok] = next_special_id
        token_strings[next_special_id] = tok
        next_special_id += 1
    
    # Process merges to build token strings
    for (a, b), new_id in merges:
        str_a = token_strings.get(a, f"<{a}>")
        str_b = token_strings.get(b, f"<{b}>")
        merged_str = str_a + str_b
        token_strings[new_id] = merged_str
        
        # Add to vocabulary if not already present
        if merged_str not in vocab:
            vocab[merged_str] = new_id
    
    # Convert to list for compatibility
    vocab_list = list(vocab.keys())
    
    stats = {
        'train_time': train_time,
        'merges_count': len(merges),
        'vocab_size': len(vocab_list),
        'corpus_size': len(corpus_bytes)
    }
    
    return vocab_list, merges, stats


def main():
    print("=" * 70)
    print("  CRAYON HYPER-FAST CODE PROFILE TRAINER")
    print("  Algorithm: Linked-List + Inverted Index + Lazy Heap BPE")
    print("  Training on: CRAYON_Full_Codebase.txt")
    print("=" * 70)
    
    # Check corpus file exists
    if not CODEBASE_FILE.exists():
        print(f"\nERROR: Corpus file not found: {CODEBASE_FILE}")
        print("Please run export_codebase.py first and move the file to resources/")
        return
    
    # Load corpus as bytes (required for C++ trainer)
    print("\n[1/6] Loading corpus...")
    start_load = time.perf_counter()
    
    with open(CODEBASE_FILE, 'rb') as f:
        corpus_bytes = f.read()
    
    # Also load as text for stats
    corpus_text = corpus_bytes.decode('utf-8', errors='replace')
    
    load_time = time.perf_counter() - start_load
    corpus_lines = corpus_text.count('\n')
    corpus_chars = len(corpus_text)
    
    print(f"       Corpus: {CODEBASE_FILE.name}")
    print(f"       Size:   {format_size(len(corpus_bytes))}")
    print(f"       Lines:  {corpus_lines:,}")
    print(f"       Chars:  {corpus_chars:,}")
    print(f"       Loaded in {format_time(load_time)}")
    
    # ========================================
    # BEFORE: Load current vocabulary
    # ========================================
    print("\n[2/6] Analyzing current vocabulary...")
    
    before_vocab = load_vocab(CURRENT_JSON)
    before_stats = analyze_vocab(before_vocab, "Current 'code' Profile")
    
    # Backup current vocab if it exists
    if before_vocab:
        backup_path = RESOURCES_DIR / "vocab_code_backup.json"
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(before_vocab, f, indent=2, ensure_ascii=False)
        print(f"\n  Backup saved: {backup_path.name}")
    
    # ========================================
    # TRAINING: Build new vocabulary with C++ engine
    # ========================================
    print("\n[3/6] Training vocabulary with C++ engine...")
    print("=" * 60)
    
    print("\n  Training parameters:")
    print("    Target size:    250,000 tokens")
    print("    Min frequency:  2")
    print("    Max token len:  16 bytes (SIMD constraint)")
    print()
    
    try:
        new_vocab, merges, train_stats = train_with_cpp_engine(
            corpus_bytes,
            vocab_size=250000,
            min_freq=2
        )
    except ImportError as e:
        print(f"\n  ERROR: C++ trainer not available!")
        print(f"  Run: pip install -e . --no-build-isolation")
        print(f"  Exception: {e}")
        return
    
    print(f"\n  Training complete!")
    print(f"    Merge rules: {train_stats['merges_count']:,}")
    print(f"    Vocab size:  {train_stats['vocab_size']:,} tokens")
    print(f"    Train time:  {format_time(train_stats['train_time'])}")
    print(f"    Speed:       {len(corpus_bytes) / train_stats['train_time'] / (1024*1024):.2f} MB/s")
    print(f"    Merges/sec:  {train_stats['merges_count'] / train_stats['train_time']:,.0f}")
    
    # ========================================
    # SAVE: Write vocabulary JSON
    # ========================================
    print("\n[4/6] Saving vocabulary...")
    
    # Create structured output
    output_data = {
        "metadata": {
            "version": "2.0.0",
            "algorithm": "Linked-List + Inverted Index + Lazy Heap BPE",
            "corpus_file": CODEBASE_FILE.name,
            "corpus_size": len(corpus_bytes),
            "vocab_size": len(new_vocab),
            "merges_count": len(merges),
            "min_freq": 2,
            "train_time_seconds": train_stats['train_time'],
            "created_at": datetime.now().isoformat()
        },
        "vocab": {token: idx for idx, token in enumerate(new_vocab)},
        "merges": [
            {"pair": [a, b], "new_id": new_id}
            for (a, b), new_id in merges
        ]
    }
    
    # Ensure directory exists
    RESOURCES_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(NEW_JSON, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    json_size = NEW_JSON.stat().st_size
    print(f"       Saved JSON: {NEW_JSON.name} ({format_size(json_size)})")
    
    # ========================================
    # BUILD DAT: Compile for runtime
    # ========================================
    print("\n[5/6] Building DAT file (C++ Accelerated)...")
    
    try:
        from crayon.c_ext import crayon_compiler
        print(f"       Compiler Version: {crayon_compiler.get_version()}")
        
        start_dat = time.perf_counter()
        
        # ONE LINE TO RULE THEM ALL
        # Passes the list of strings and the output path directly to C++
        stats = crayon_compiler.compile_dat(new_vocab, str(NEW_DAT))
        
        dat_time = time.perf_counter() - start_dat
        
        dat_size = NEW_DAT.stat().st_size
        print(f"       Saved DAT:  {NEW_DAT.name} ({format_size(dat_size)})")
        print(f"       DAT nodes:  {stats.get('node_count', 0):,}")
        print(f"       Build time: {format_time(dat_time)}")
        
    except ImportError:
        print("\n  ⚠️  FAST COMPILER NOT FOUND!")
        print("  Falling back to slow Python builder (this will take a while...)")
        
        start_dat = time.perf_counter()
        builder = DATBuilder()
        builder.build(new_vocab)
        builder.save(str(NEW_DAT))
        dat_time = time.perf_counter() - start_dat
        
        dat_size = NEW_DAT.stat().st_size
        print(f"       Saved DAT:  {NEW_DAT.name} ({format_size(dat_size)})")
        print(f"       DAT nodes:  {builder.node_count:,}")
        print(f"       Build time: {format_time(dat_time)}")

    
    # ========================================
    # AFTER: Analyze new vocabulary
    # ========================================
    print("\n[6/6] Analyzing new vocabulary...")
    
    after_stats = analyze_vocab(new_vocab, "New 'code' Profile (trained on CRAYON codebase)")
    
    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "#" * 70)
    print("  BEFORE vs AFTER COMPARISON")
    print("#" * 70)
    
    if before_stats and after_stats:
        print("\n  Metric               Before      After      Change")
        print("  " + "-" * 55)
        
        metrics = [
            ('Total tokens', 'total'),
            ('Special tokens', 'special'),
            ('Single chars', 'single'),
            ('Multi-char tokens', 'multi'),
            ('Python keywords', 'py_keywords'),
            ('C/C++ keywords', 'cpp_keywords'),
            ('CUDA keywords', 'cuda_keywords'),
        ]
        
        for label, key in metrics:
            before = before_stats.get(key, 0)
            after = after_stats.get(key, 0)
            change = after - before
            sign = '+' if change > 0 else ''
            print(f"  {label:20s} {before:>8,}   {after:>8,}   {sign}{change:,}")
        
        # Average length
        before_len = before_stats.get('avg_len', 0)
        after_len = after_stats.get('avg_len', 0)
        print(f"  {'Avg token length':20s} {before_len:>8.2f}   {after_len:>8.2f}   {after_len - before_len:+.2f}")
    else:
        print("\n  (No comparison available - before was empty)")
    
    # ========================================
    # SUMMARY
    # ========================================
    total_time = time.perf_counter() - start_load
    
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\n  Performance Summary:")
    print(f"    Corpus Size:    {format_size(len(corpus_bytes))}")
    print(f"    Vocab Size:     {len(new_vocab):,} tokens")
    print(f"    Merge Rules:    {len(merges):,}")
    print(f"    Train Time:     {format_time(train_stats['train_time'])}")
    print(f"    Total Time:     {format_time(total_time)}")
    print(f"    Train Speed:    {len(corpus_bytes) / train_stats['train_time'] / (1024*1024):.2f} MB/s")
    
    print(f"\n  Output Files:")
    print(f"    JSON: {NEW_JSON}")
    print(f"    DAT:  {NEW_DAT}")
    
    print(f"\n  To use: vocab.load_profile('code')")
    print("=" * 70)


if __name__ == "__main__":
    main()
