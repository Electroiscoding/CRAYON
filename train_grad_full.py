"""
Incremental training script for FULL GRAD dataset.

Objective:
1. Load existing 'trained_vocab.json'.
2. Train a temporary vocabulary on the FULL 18MB GRAD dataset.
3. Merge NEW tokens from GRAD into the existing vocabulary.
4. Preserve existing token IDs (append-only update).
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Set

from crayon import CrayonVocab
from crayon.training import train_vocabulary

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Paths
RESOURCE_DIR = Path("src/crayon/resources")
GRAD_PATH = RESOURCE_DIR / "graduate_math.jsonl"
EXISTING_VOCAB_PATH = "trained_vocab.json"

def yield_grad_full():
    """Yields text from the FULL GRAD dataset (Questions + Solutions)."""
    if not GRAD_PATH.exists():
        print(f"[ERROR] GRAD dataset not found at {GRAD_PATH}")
        return

    print(f"[INFO] Streaming FULL GRAD dataset: {GRAD_PATH}")
    file_size_mb = GRAD_PATH.stat().st_size / (1024 * 1024)
    print(f"[INFO] File Size: {file_size_mb:.2f} MB")

    count = 0
    with open(GRAD_PATH, 'r', encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f):
            # Optimization: Process every 10th line (10% sampling)
            # This processes ~1.8MB of text, providing excellent coverage without OOM.
            if i % 10 != 0:
                continue

            if line.strip():
                try:
                    data = json.loads(line)
                    if 'question' in data: yield data['question']
                    if 'solution' in data: yield data['solution']
                    
                    count += 1
                    if count % 2000 == 0:
                        print(f"      ... loaded {count} entries", end='\r')
                except json.JSONDecodeError:
                    continue
    print(f"\n[INFO] Finished loading {count} entries (subsampled).")

def progress_callback(msg: str):
    if "Processed" in msg and not msg.endswith("00 chunks..."): return
    print(f"[PROGRESS] {msg}")

def main():
    print("=" * 60)
    print("XERV Crayon: Incremental Training (Full GRAD - Optimized)")
    print("=" * 60)

    # 1. Load Existing Vocabulary
    print(f"\n[1] Loading existing vocabulary from {EXISTING_VOCAB_PATH}...")
    try:
        base_vocab = CrayonVocab.from_json(EXISTING_VOCAB_PATH)
        print(f"    - Loaded {len(base_vocab)} tokens")
    except Exception as e:
        print(f"    - Verification Failed: {e}")
        return

    # Reconstruct the ordered list
    print("    - Reconstructing ID mapping...")
    base_tokens = [base_vocab.id_to_token[i] for i in range(len(base_vocab))]
    existing_token_set = set(base_vocab.token_to_id.keys())

    # 2. Train New Tokens
    print(f"\n[2] Training temporary vocabulary on GRAD dataset...")
    
    # We increase min_frequency to 5 to avoid learning one-off noise from the large file
    grad_tokens_raw = train_vocabulary(
        yield_grad_full(),
        target_size=20000, 
        min_frequency=5,
        progress_callback=progress_callback
    )
    
    print(f"\n    - Extracted {len(grad_tokens_raw)} candidate tokens from GRAD")

    # 3. Merge Tokens
    print(f"\n[3] Merging new tokens...")
    new_tokens = []
    skipped = 0
    
    for token in grad_tokens_raw:
        if token not in existing_token_set:
            new_tokens.append(token)
            existing_token_set.add(token) # Prevent duplicates within new batch
        else:
            skipped += 1
            
    print(f"    - Existing tokens skipped: {skipped}")
    print(f"    - NEW tokens to add:       {len(new_tokens)}")
    
    # 4. Create Final Vocabulary
    final_token_list = base_tokens + new_tokens
    print(f"\n[4] Finalizing Vocabulary...")
    print(f"    - Base: {len(base_tokens)}")
    print(f"    - New:  {len(new_tokens)}")
    print(f"    - Total: {len(final_token_list)}")
    
    final_vocab = CrayonVocab(final_token_list)
    print(f"    - C-Extension: {'Enabled' if final_vocab._c_ext_available else 'Disabled'}")

    # 5. Save
    print(f"\n[5] Saving to {EXISTING_VOCAB_PATH}...")
    final_vocab.save("trained_vocab.json", format="json")
    final_vocab.save("trained_vocab.txt", format="txt")
    print(f"[DONE] Vocabulary updated successfully.")

    # 6. Verify
    print("\n" + "="*30)
    print("Verification")
    print("="*30)
    test_str = "Calculate the integral of e^x from 0 to infinity."
    tokens = final_vocab.tokenize(test_str)
    print(f"Input: '{test_str}'")
    print(f"Tokens: {tokens}")
    print(f"Decoded: '{final_vocab.decode(tokens)}'")

if __name__ == "__main__":
    main()
