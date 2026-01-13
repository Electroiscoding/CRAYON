"""
Background HuggingFace Dataset Training Script.

Downloads and trains CRAYON vocabulary on famous code datasets from HuggingFace Hub.
Designed to run in background with progress logging to file.

Datasets:
1. bigcode/starcoderdata (Starcoder training data - Python subset)
2. codeparrot/github-code (GitHub code samples)
3. sahil2801/CodeAlpaca-20k (Code instruction pairs)
4. m-a-p/CodeFeedback-Filtered-Instruction (Code feedback)
5. iamtarun/python_code_instructions_18k_alpaca (Python instructions)

Usage:
    python train_hf_datasets.py

Output:
    - Updates trained_vocab.json with new tokens
    - Logs progress to hf_training.log
"""

import json
import time
import logging
import sys
import os
from pathlib import Path
from typing import Iterator, Set, List, Optional
from datetime import datetime

# Set environment variable to suppress symlink warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# Configure logging to both file and console
log_file = Path("hf_training.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import datasets library
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
    logger.info("HuggingFace datasets library loaded successfully")
except ImportError:
    HF_AVAILABLE = False
    logger.error("HuggingFace datasets not installed. Run: pip install datasets")
    sys.exit(1)

from crayon import CrayonVocab
from crayon.training import train_vocabulary

# ============================================================================
# Configuration
# ============================================================================

EXISTING_VOCAB_PATH = Path("trained_vocab.json")

# Reliable HuggingFace datasets that work well with streaming
# Format: (name, config, split, text_fields, sample_size, description)
HF_DATASETS = [
    {
        "name": "sahil2801/CodeAlpaca-20k",
        "config": None,
        "split": "train",
        "text_fields": ["instruction", "input", "output"],
        "sample_size": 20000,
        "description": "CodeAlpaca instruction-following dataset"
    },
    {
        "name": "iamtarun/python_code_instructions_18k_alpaca",
        "config": None,
        "split": "train",
        "text_fields": ["instruction", "input", "output"],
        "sample_size": 18000,
        "description": "Python code instructions dataset"
    },
    {
        "name": "m-a-p/CodeFeedback-Filtered-Instruction",
        "config": None,
        "split": "train",
        "text_fields": ["query", "answer"],
        "sample_size": 15000,
        "description": "Code feedback and instruction pairs"
    },
    {
        "name": "nickrosh/Evol-Instruct-Code-80k-v1",
        "config": None,
        "split": "train",
        "text_fields": ["instruction", "output"],
        "sample_size": 20000,
        "description": "Evolved code instructions (80k samples)"
    },
    {
        "name": "theblackcat102/evol-codealpaca-v1",
        "config": None,
        "split": "train",
        "text_fields": ["instruction", "output"],
        "sample_size": 15000,
        "description": "Evolved CodeAlpaca dataset"
    },
    {
        "name": "TokenBender/code_instructions_122k_alpaca_style",
        "config": None,
        "split": "train",
        "text_fields": ["instruction", "input", "output"],
        "sample_size": 25000,
        "description": "Large code instructions dataset (122k)"
    },
    {
        "name": "flytech/python-codes-25k",
        "config": None,
        "split": "train",
        "text_fields": ["text", "code"],
        "sample_size": 25000,
        "description": "Python code samples (25k)"
    },
    {
        "name": "Vezora/Tested-143k-Python-Alpaca",
        "config": None,
        "split": "train",
        "text_fields": ["instruction", "input", "output"],
        "sample_size": 30000,
        "description": "Tested Python code samples"
    },
]


def stream_hf_dataset(config: dict) -> Iterator[str]:
    """
    Streams text from a HuggingFace dataset.
    
    Args:
        config: Dataset configuration dict
        
    Yields:
        Text chunks from the dataset
    """
    name = config["name"]
    subset = config.get("config")
    split = config.get("split", "train")
    text_fields = config["text_fields"]
    sample_size = config.get("sample_size", 10000)
    description = config.get("description", name)
    
    logger.info(f"Loading: {name} ({description})")
    logger.info(f"  Target samples: {sample_size:,}")
    
    try:
        # Load dataset with streaming for memory efficiency
        if subset:
            dataset = load_dataset(name, subset, split=split, streaming=True)
        else:
            dataset = load_dataset(name, split=split, streaming=True)
        
        count = 0
        for example in dataset:
            if count >= sample_size:
                break
            
            # Extract text from all specified fields
            for field in text_fields:
                if field in example:
                    text = example[field]
                    if text and isinstance(text, str) and len(text) > 10:
                        yield text
                        count += 1
                        
                        if count % 5000 == 0:
                            logger.info(f"  {name}: {count:,}/{sample_size:,} samples loaded...")
                        
                        if count >= sample_size:
                            break
        
        logger.info(f"  Completed: {count:,} samples from {name}")
        return
            
    except Exception as e:
        logger.error(f"  FAILED to load {name}: {str(e)[:100]}")
        return


def yield_all_hf_datasets() -> Iterator[str]:
    """
    Yields text from ALL configured HuggingFace datasets.
    """
    total_yielded = 0
    successful_datasets = 0
    failed_datasets = 0
    
    logger.info("=" * 60)
    logger.info("Starting HuggingFace Dataset Download and Processing")
    logger.info("=" * 60)
    logger.info(f"Total datasets to process: {len(HF_DATASETS)}")
    logger.info("")
    
    for i, config in enumerate(HF_DATASETS, 1):
        logger.info(f"[{i}/{len(HF_DATASETS)}] Processing: {config['name']}")
        
        try:
            dataset_count = 0
            for text in stream_hf_dataset(config):
                yield text
                total_yielded += 1
                dataset_count += 1
            
            if dataset_count > 0:
                successful_datasets += 1
            else:
                failed_datasets += 1
                
        except Exception as e:
            logger.error(f"  Error processing {config['name']}: {e}")
            failed_datasets += 1
        
        logger.info("")
    
    logger.info("=" * 60)
    logger.info("HuggingFace Dataset Processing Complete")
    logger.info(f"  Successful datasets: {successful_datasets}")
    logger.info(f"  Failed datasets: {failed_datasets}")
    logger.info(f"  Total samples yielded: {total_yielded:,}")
    logger.info("=" * 60)


def main():
    start_time = datetime.now()
    
    logger.info("=" * 70)
    logger.info("XERV Crayon: HuggingFace Dataset Training")
    logger.info(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    logger.info("")
    
    # 1. Load Existing Vocabulary
    logger.info(f"[1] Loading existing vocabulary from {EXISTING_VOCAB_PATH}...")
    
    if not EXISTING_VOCAB_PATH.exists():
        logger.error(f"    {EXISTING_VOCAB_PATH} not found!")
        logger.error("    Run train_vocab.py first to create base vocabulary.")
        return
    
    try:
        base_vocab = CrayonVocab.from_json(str(EXISTING_VOCAB_PATH))
        base_size = len(base_vocab)
        logger.info(f"    Loaded {base_size:,} tokens")
        logger.info(f"    C-Extension: {'Enabled' if base_vocab._c_ext_available else 'Disabled'}")
    except Exception as e:
        logger.error(f"    Failed to load vocabulary: {e}")
        return
    
    # Reconstruct ordered token list and set for O(1) lookup
    logger.info("    Reconstructing ID mapping...")
    base_tokens = [base_vocab.id_to_token[i] for i in range(len(base_vocab))]
    existing_token_set = set(base_vocab.token_to_id.keys())
    
    # 2. Download and Train on HuggingFace Datasets
    logger.info("")
    logger.info("[2] Downloading and processing HuggingFace datasets...")
    logger.info("    This may take 10-30 minutes depending on network speed.")
    logger.info("")
    
    def progress_callback(msg: str):
        if "Processed" in msg and not msg.endswith("00 chunks..."):
            return
        logger.info(f"[TRAIN] {msg}")
    
    train_start = time.time()
    
    # Train vocabulary on HF data
    hf_tokens_raw = train_vocabulary(
        yield_all_hf_datasets(),
        target_size=50000,  # Extract up to 50k code tokens
        min_frequency=3,    # Require at least 3 occurrences
        progress_callback=progress_callback
    )
    
    training_time = time.time() - train_start
    logger.info("")
    logger.info(f"    Extracted {len(hf_tokens_raw):,} candidate tokens in {training_time:.1f}s")
    
    # 3. Merge Tokens (Append-Only, ID-Stable)
    logger.info("")
    logger.info("[3] Merging new tokens (append-only)...")
    
    new_tokens = []
    skipped = 0
    
    for token in hf_tokens_raw:
        if token not in existing_token_set:
            new_tokens.append(token)
            existing_token_set.add(token)  # Prevent duplicates within batch
        else:
            skipped += 1
    
    logger.info(f"    Existing tokens skipped: {skipped:,}")
    logger.info(f"    NEW tokens to add:       {len(new_tokens):,}")
    
    # Show sample of new tokens
    if new_tokens:
        logger.info("")
        logger.info("    Sample new tokens (first 20):")
        for i, token in enumerate(new_tokens[:20]):
            display = repr(token) if len(token) < 25 else repr(token[:22] + "...")
            logger.info(f"      [{i:2d}] {display}")
    
    # 4. Create Final Vocabulary
    logger.info("")
    logger.info("[4] Creating final vocabulary...")
    final_token_list = base_tokens + new_tokens
    
    logger.info(f"    Base vocabulary:  {len(base_tokens):,}")
    logger.info(f"    New HF tokens:    {len(new_tokens):,}")
    logger.info(f"    Total vocabulary: {len(final_token_list):,}")
    
    final_vocab = CrayonVocab(final_token_list)
    logger.info(f"    C-Extension: {'Enabled' if final_vocab._c_ext_available else 'Disabled'}")
    
    # 5. Save Updated Vocabulary
    logger.info("")
    logger.info(f"[5] Saving to {EXISTING_VOCAB_PATH}...")
    final_vocab.save(str(EXISTING_VOCAB_PATH), format="json")
    final_vocab.save("trained_vocab.txt", format="txt")
    logger.info("    Vocabulary updated successfully!")
    
    # 6. Verification
    logger.info("")
    logger.info("=" * 60)
    logger.info("Verification Tests")
    logger.info("=" * 60)
    
    test_cases = [
        ("Python Function", "def calculate_sum(a: int, b: int) -> int:\n    return a + b"),
        ("Python Class", "class DataLoader:\n    def __init__(self, path):\n        self.path = path"),
        ("JavaScript", "const fetchData = async (url) => await fetch(url).then(r => r.json())"),
        ("TypeScript", "interface Config { apiKey: string; timeout: number; }"),
        ("Code Comment", "# This function calculates the factorial of a number recursively"),
    ]
    
    for lang, test_str in test_cases:
        tokens = final_vocab.tokenize(test_str)
        decoded = final_vocab.decode(tokens)
        match = "[OK]" if decoded == test_str else "[DIFF]"
        
        display = test_str[:45] + "..." if len(test_str) > 45 else test_str
        display = display.replace('\n', '\\n')
        logger.info(f"  [{lang}] {match} - {len(tokens)} tokens")
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Original vocabulary: {base_size:,} tokens")
    logger.info(f"  Final vocabulary:    {len(final_vocab):,} tokens")
    logger.info(f"  New tokens added:    {len(new_tokens):,}")
    logger.info(f"  Training time:       {training_time:.1f}s")
    logger.info(f"  Total duration:      {duration}")
    logger.info(f"  Output file:         {EXISTING_VOCAB_PATH}")
    logger.info(f"  Log file:            {log_file}")
    logger.info("")
    
    # Write summary to a separate file
    summary_file = Path("hf_training_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"XERV Crayon HuggingFace Training Summary\n")
        f.write(f"{'=' * 50}\n")
        f.write(f"Started:     {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Completed:   {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration:    {duration}\n")
        f.write(f"\n")
        f.write(f"Original vocabulary: {base_size:,} tokens\n")
        f.write(f"Final vocabulary:    {len(final_vocab):,} tokens\n")
        f.write(f"New tokens added:    {len(new_tokens):,}\n")
        f.write(f"\n")
        f.write(f"Datasets processed:\n")
        for ds in HF_DATASETS:
            f.write(f"  - {ds['name']}: {ds['sample_size']:,} samples\n")
    
    logger.info(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
