"""
Train Vocabulary - FULL GRAD DATASET ONLY.

Source: src/crayon/resources/graduate_math.jsonl
Mode: Full dataset (Questions + Solutions)
"""

import os
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
GRAD_PATH = RESOURCE_DIR / "graduate_math.jsonl"

def yield_grad_only():
    """Yields text ONLY from the full GRAD dataset."""
    
    if not GRAD_PATH.exists():
        print(f"[ERROR] file not found: {GRAD_PATH}")
        return

    print(f"[INFO] Streaming FULL GRAD dataset: {GRAD_PATH}")
    filesize = GRAD_PATH.stat().st_size
    print(f"[INFO] File Size: {filesize / 1024 / 1024:.2f} MB")

    count = 0
    with open(GRAD_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    # Yield both question and solution for maximum math/logic coverage
                    if 'question' in data:
                        yield data['question']
                    if 'solution' in data:
                        yield data['solution']
                    count += 1
                    if count % 1000 == 0:
                        print(f"      ... loaded {count} entries", end='\r')
                except json.JSONDecodeError:
                    continue
    print(f"\n[INFO] Finished loading {count} entries.")


def progress_callback(msg: str):
    print(f"[PROGRESS] {msg}")


def main():
    print("=" * 60)
    print("XERV Crayon Training: FULL GRAD DATASET")
    print("=" * 60)
    
    start_time = time.time()
    
    # Build vocabulary from local corpus
    corpus_iter = yield_grad_only()
    
    # Train vocabulary
    # We use a slightly smaller vocab size (32k) for strictly math/specialized domains 
    # to avoid overfitting noise, or keep 50k if the user wants "max capacity".
    # Defaulting to 50k as per previous.
    tokens = train_vocabulary(
        corpus_iter,
        target_size=50000,
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n[DONE] Vocabulary built in {elapsed:.1f}s")
    print(f"       Token count: {len(tokens)}")
    
    # Create CrayonVocab
    vocab = CrayonVocab(tokens)
    print(f"       C-Extension: {'Enabled' if vocab._c_ext_available else 'Disabled'}")
    
    # Save
    vocab.save("trained_vocab.json", format="json")
    vocab.save("trained_vocab.txt", format="txt")
    print(f"\n[SAVED] trained_vocab.json")
    
    # Verify on a math-heavy string
    test_str = "Calculate the integral of e^x from 0 to infinity."
    tokens = vocab.tokenize(test_str)
    print(f"\n[TEST]: '{test_str}'")
    print(f"Tokens: {tokens}")
    print(f"Decode: '{vocab.decode(tokens)}'")

if __name__ == "__main__":
    main()
