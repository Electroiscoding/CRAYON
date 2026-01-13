"""
Train Vocabulary from Local Resources.

Uses local files from src/crayon/resources/:
- input.txt (Shakespeare)
- data.csv (RainDrop-DTS)
- physics_detailed_dataset_700_rows.csv (Physics)
- graduate_math.jsonl (GRAD)
"""

import os
import csv
import json
import time
import logging
from pathlib import Path
from crayon import CrayonVocab
from crayon.training import train_vocabulary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Resource directory
RESOURCE_DIR = Path(__file__).parent / "src" / "crayon" / "resources"


def yield_local_corpus():
    """Yields text from all local resource files."""
    
    # 1. Shakespeare (input.txt)
    shakespeare_path = RESOURCE_DIR / "input.txt"
    if shakespeare_path.exists():
        print(f"[INFO] Loading Shakespeare from {shakespeare_path}")
        with open(shakespeare_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield line.strip()
    
    # 2. RainDrop-DTS (data.csv)
    raindrop_path = RESOURCE_DIR / "data.csv"
    if raindrop_path.exists():
        print(f"[INFO] Loading RainDrop-DTS from {raindrop_path}")
        with open(raindrop_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'text' in row and row['text']:
                    yield row['text']
    
    # 3. Physics dataset
    physics_path = RESOURCE_DIR / "physics_detailed_dataset_700_rows.csv"
    if physics_path.exists():
        print(f"[INFO] Loading Physics from {physics_path}")
        with open(physics_path, 'r', encoding='utf-8', errors='ignore') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for col in ['Question', 'Answer', 'Reasoning']:
                    if col in row and row[col]:
                        yield row[col]
    
    # 4. GRAD (graduate_math.jsonl)
    grad_path = RESOURCE_DIR / "graduate_math.jsonl"
    if grad_path.exists():
        print(f"[INFO] Loading GRAD from {grad_path}")
        with open(grad_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'question' in data:
                            yield data['question']
                        if 'solution' in data:
                            yield data['solution']
                    except json.JSONDecodeError:
                        continue


def progress_callback(msg: str):
    print(f"[PROGRESS] {msg}")


def main():
    print("=" * 60)
    print("XERV Crayon Vocabulary Training (Local Resources)")
    print("=" * 60)
    
    print(f"\nResource directory: {RESOURCE_DIR}")
    print("\nDatasets:")
    for f in RESOURCE_DIR.iterdir():
        print(f"  - {f.name} ({f.stat().st_size / 1024:.1f} KB)")
    print()
    
    start_time = time.time()
    
    # Build vocabulary from local corpus
    corpus_iter = yield_local_corpus()
    tokens = train_vocabulary(
        corpus_iter,
        target_size=50000,
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n[DONE] Vocabulary built in {elapsed:.1f}s")
    print(f"       Token count: {len(tokens)}")
    
    # Create CrayonVocab with the tokens
    vocab = CrayonVocab(tokens)
    print(f"       C-Extension: {'Enabled' if vocab._c_ext_available else 'Disabled'}")
    
    # Save vocabulary
    vocab.save("trained_vocab.json", format="json")
    vocab.save("trained_vocab.txt", format="txt")
    print(f"\n[SAVED] trained_vocab.json")
    print(f"[SAVED] trained_vocab.txt")
    
    # Test it
    print("\n" + "=" * 60)
    print("Testing Trained Vocabulary")
    print("=" * 60)
    
    test_texts = [
        "delhi is the capital of india",
        "The quick brown fox jumps over the lazy dog",
        "What is the derivative of x squared?",
        "Calculate the force using F = ma",
    ]
    
    for text in test_texts:
        tokens = vocab.tokenize(text)
        decoded = vocab.decode(tokens)
        unk_count = tokens.count(vocab.unk_token_id)
        print(f"\nInput:   '{text}'")
        print(f"Tokens:  {len(tokens)} total, {unk_count} unknown")
        print(f"Decoded: '{decoded}'")


if __name__ == "__main__":
    main()
