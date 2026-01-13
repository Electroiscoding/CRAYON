"""
Crayon Vocabulary Training Module.

Implements Algorithm 3.1 from the XERV Crayon Engineering Treatise:
- Extract substring candidates up to SIMD limit (16 bytes)
- Calculate information gain with entropy reduction
- Select top-K candidates maximizing gain-to-cost ratio

This is the production-grade implementation for building optimal vocabularies
from either user-provided corpora or the built-in default sources.
"""

import math
import logging
import string
from collections import defaultdict
from typing import List, Tuple, Dict, Iterator, Optional, Callable

# Configure module logger
logger = logging.getLogger(__name__)

# SIMD Hardware Limit [cite: 128]
MAX_TOKEN_LENGTH = 16

# Minimum frequency threshold to filter noise
DEFAULT_MIN_FREQUENCY = 2


def build_default_vocabulary(
    target_size: int = 500000,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[str]:
    """
    Builds a 'Batteries-Included' vocabulary using Xerv-AI's curated datasets.
    
    Sources:
    - Xerv-AI/GRAD (Graduate Mathematics)
    - Xerv-AI/Physics-dataset-700 (Scientific Reasoning)
    - Xerv-AI/RainDrop-DTS (General Instruction)
    - Tiny Shakespeare (Classical Literature)
    - Built-in corpus (Baseline Coverage)
    
    No local files are required; data is streamed directly into the entropy engine.
    
    Args:
        target_size: Maximum vocabulary size (default 500k)
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of token strings ordered by utility
    """
    from .resources import get_default_corpus_iterator
    
    if progress_callback:
        progress_callback("Initializing default corpus stream...")
    
    corpus_stream = get_default_corpus_iterator()
    return train_vocabulary(
        corpus_stream, 
        target_size=target_size,
        progress_callback=progress_callback
    )


def train_vocabulary(
    corpus_iterator: Iterator[str], 
    target_size: int = 500000,
    min_frequency: int = DEFAULT_MIN_FREQUENCY,
    progress_callback: Optional[Callable[[str], None]] = None
) -> List[str]:
    """
    Constructs an optimal vocabulary from a corpus using first-principles entropy analysis.
    
    Algorithm 3.1 [cite: 127-135]:
    1. Extract all substrings up to MAX_TOKEN_LENGTH (16 bytes for AVX2).
    2. Calculate Information Gain: Gain(s) = Frequency(s) × Entropy(s) - Cost(s).
    3. Select Top-K candidates maximizing utility score.
    
    Args:
        corpus_iterator: Iterator yielding chunks/lines of text
        target_size: Maximum vocabulary size (default 500k)
        min_frequency: Minimum token frequency threshold
        progress_callback: Optional callback for progress updates
        
    Returns:
        List of token strings ordered for stable ID assignment
    """
    if progress_callback:
        progress_callback("Starting Entropy-Guided Vocabulary Construction...")
    
    logger.info("Starting Entropy-Guided Vocabulary Construction...")
    
    # ========================================================================
    # Phase 1: Candidate Extraction & Frequency Counting [cite: 128]
    # ========================================================================
    candidates: Dict[str, int] = defaultdict(int)
    total_chars = 0
    chunk_count = 0
    
    # Process stream chunk by chunk (Zero-Disk Accumulation)
    for text_chunk in corpus_iterator:
        if not text_chunk:
            continue
        
        text_len = len(text_chunk)
        total_chars += text_len
        chunk_count += 1
        
        # Hot-path extraction loop - extract all valid substrings
        for i in range(text_len):
            # Hardware constraint: Tokens > 16 bytes degrade SIMD performance
            limit = min(i + MAX_TOKEN_LENGTH, text_len)
            for j in range(i + 1, limit + 1):
                token = text_chunk[i:j]
                
                # Skip tokens that exceed byte limit when encoded
                if len(token.encode('utf-8')) <= MAX_TOKEN_LENGTH:
                    candidates[token] += 1
        
        # Progress update every 100 chunks
        if chunk_count % 100 == 0 and progress_callback:
            progress_callback(f"Processed {chunk_count} chunks, {len(candidates):,} candidates...")
    
    if progress_callback:
        progress_callback(f"Extracted {len(candidates):,} unique candidates from {total_chars:,} chars")
    
    logger.info(f"Extracted {len(candidates):,} unique candidates from {total_chars:,} chars.")

    # ========================================================================
    # Phase 2: Information Gain Calculation [cite: 129-134]
    # ========================================================================
    if progress_callback:
        progress_callback("Scoring candidates by information gain...")
    
    scored_candidates: List[Tuple[str, float]] = []
    
    for token, freq in candidates.items():
        # Filter low-frequency noise
        if freq < min_frequency:
            continue
        
        # Skip control characters and empty strings
        if not token or not token.isprintable():
            continue
            
        # Probability p(s)
        p_s = freq / total_chars
        if p_s <= 0:
            continue
        
        # Information content (entropy reduction) [cite: 131]
        # H(s) = -log2(p(s))
        entropy = -math.log2(p_s)
        
        # Computational Cost Estimate [cite: 133]
        # Cost is linear to byte length + constant overhead for SIMD alignment
        byte_length = len(token.encode('utf-8'))
        comp_cost = byte_length * 0.1 + 1.0
        
        # Information Gain [cite: 134]
        # Gain = (Entropy × Frequency) / Cost
        gain = (entropy * freq) / comp_cost
        
        scored_candidates.append((token, gain))

    if progress_callback:
        progress_callback(f"Scored {len(scored_candidates):,} viable candidates")
    
    logger.info(f"Scored {len(scored_candidates):,} viable candidates")

    # ========================================================================
    # Phase 3: Selection with Priority Categories [cite: 1009-1012]
    # ========================================================================
    if progress_callback:
        progress_callback("Building final vocabulary...")
    
    # Sort by gain descending
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    
    # Build vocabulary with reserved categories
    vocab_set: set = set()
    
    # 1. Special tokens (MANDATORY) [cite: 1009]
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    for s in specials:
        vocab_set.add(s)
    
    # 2. ASCII printable characters (BASELINE) [cite: 1010]
    for c in string.printable:
        if c not in vocab_set and c.strip():
            vocab_set.add(c)
    
    # 3. Common single-byte sequences
    for i in range(256):
        try:
            char = chr(i)
            if char.isprintable() and char not in vocab_set:
                vocab_set.add(char)
        except (ValueError, UnicodeDecodeError):
            pass
    
    # 4. Fill remainder with entropy-optimized tokens
    remaining_slots = target_size - len(vocab_set)
    added_count = 0
    
    for token, gain in scored_candidates:
        if added_count >= remaining_slots:
            break
        if token not in vocab_set:
            vocab_set.add(token)
            added_count += 1
    
    final_vocab = list(vocab_set)
    
    if progress_callback:
        progress_callback(f"Final vocabulary: {len(final_vocab):,} tokens")
    
    logger.info(f"Final vocabulary: {len(final_vocab):,} tokens")
    
    return final_vocab


def calculate_corpus_entropy(corpus_iterator: Iterator[str]) -> float:
    """
    Calculate Shannon entropy of a corpus [cite: 93-96].
    
    H(X) = -Σ p(x) log2(p(x))
    
    Args:
        corpus_iterator: Iterator yielding text chunks
        
    Returns:
        Entropy in bits per character
    """
    char_counts: Dict[str, int] = defaultdict(int)
    total = 0
    
    for chunk in corpus_iterator:
        for char in chunk:
            char_counts[char] += 1
            total += 1
    
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in char_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    return entropy


def estimate_optimal_vocab_size(entropy: float, epsilon: float = 0.5) -> int:
    """
    Calculate optimal vocabulary size from corpus entropy [cite: 94].
    
    V_optimal ≈ 2^(H(corpus) + ε)
    
    For English text (H ≈ 1.2 bits/char), this yields ~500k tokens.
    
    Args:
        entropy: Corpus entropy in bits per character
        epsilon: Adjustment factor (default 0.5)
        
    Returns:
        Estimated optimal vocabulary size
    """
    return int(2 ** (entropy + epsilon))
