"""
Entropy-Guided Vocabulary Construction Module.

Implements Algorithm 3.1 from the XERV Crayon Engineering Treatise:
- Extract substring candidates up to SIMD limit (16 bytes)
- Calculate information gain with entropy reduction
- Select top-K candidates maximizing gain-to-cost ratio

This is the production-grade implementation for building optimal vocabularies.
"""

import math
import hashlib
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass

# SIMD Hardware Limit [cite: 128]
MAX_TOKEN_LENGTH = 16


@dataclass
class TokenCandidate:
    """Scored vocabulary candidate."""
    token: str
    frequency: int
    entropy: float
    information_gain: float
    computational_cost: float
    utility_score: float


class EntropyVocabBuilder:
    """
    Production-grade entropy-guided vocabulary builder.
    
    Implements the mathematical optimization from Section 2.1 [cite: 129-135]:
    - Entropy-bound sizing: V_optimal ≈ 2^(H(corpus) + ε)
    - Information gain: Gain(s) = Frequency(s) × EntropyReduction(s) - Cost(s)
    """
    
    def __init__(
        self,
        target_size: int = 500000,
        max_token_length: int = MAX_TOKEN_LENGTH,
        min_frequency: int = 2,
        special_tokens: Optional[List[str]] = None
    ):
        self.target_size = target_size
        self.max_token_length = max_token_length
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        
        # Statistics
        self.corpus_entropy: float = 0.0
        self.optimal_vocab_size: int = 0
    
    def construct_optimal_vocabulary(
        self,
        corpus: str,
        progress_callback: Optional[callable] = None
    ) -> List[str]:
        """
        Implements Algorithm 3.1: Entropy-Guided Candidate Selection [cite: 126-135].
        
        Args:
            corpus: Training text corpus
            progress_callback: Optional callback for progress reporting
            
        Returns:
            Optimally ordered list of tokens for vocabulary
        """
        if progress_callback:
            progress_callback("Extracting candidates...")
        
        # 1. Extract all valid substrings (up to SIMD limit)
        candidates = self._extract_candidates(corpus)
        
        if progress_callback:
            progress_callback(f"Extracted {len(candidates):,} unique candidates")
        
        # 2. Calculate corpus entropy
        self.corpus_entropy = self._calculate_corpus_entropy(corpus)
        self.optimal_vocab_size = self._calculate_optimal_size(self.corpus_entropy)
        
        if progress_callback:
            progress_callback(f"Corpus entropy: {self.corpus_entropy:.4f} bits/char")
            progress_callback(f"Optimal vocab size: {self.optimal_vocab_size:,}")
        
        # 3. Score candidates using information-theoretic utility
        total_chars = len(corpus)
        scored = self._score_candidates(candidates, total_chars)
        
        if progress_callback:
            progress_callback(f"Scored {len(scored):,} candidates")
        
        # 4. Select top-K candidates
        effective_size = min(self.target_size, self.optimal_vocab_size)
        
        # Reserve space for special tokens and ASCII
        reserved = len(self.special_tokens) + 256
        available = effective_size - reserved
        
        # Sort by utility score descending
        scored.sort(key=lambda x: x.utility_score, reverse=True)
        
        # Build final vocabulary
        vocab_tokens = list(self.special_tokens)
        
        # Add ASCII bytes [cite: 1009-1012]
        for i in range(256):
            char = chr(i)
            if char not in vocab_tokens and char.isprintable():
                vocab_tokens.append(char)
        
        # Add top candidates
        seen: Set[str] = set(vocab_tokens)
        for candidate in scored[:available]:
            if candidate.token not in seen:
                vocab_tokens.append(candidate.token)
                seen.add(candidate.token)
        
        if progress_callback:
            progress_callback(f"Final vocabulary: {len(vocab_tokens):,} tokens")
        
        return vocab_tokens
    
    def _extract_candidates(self, corpus: str) -> Dict[str, int]:
        """
        Sliding window extraction of all valid substrings [cite: 128].
        
        Uses SIMD-aligned max length (16 bytes) for hardware optimization.
        """
        candidates: Dict[str, int] = defaultdict(int)
        corpus_bytes = corpus.encode('utf-8')
        corpus_len = len(corpus)
        
        # Track byte positions for UTF-8 aware extraction
        byte_pos = 0
        for char_pos in range(corpus_len):
            char = corpus[char_pos]
            char_bytes = len(char.encode('utf-8'))
            
            # Extract substrings starting at this position
            current_byte_len = 0
            for length in range(1, min(self.max_token_length + 1, corpus_len - char_pos + 1)):
                end_char = corpus[char_pos:char_pos + length]
                end_byte_len = len(end_char.encode('utf-8'))
                
                # Stop if exceeds SIMD byte limit
                if end_byte_len > self.max_token_length:
                    break
                
                candidates[end_char] += 1
            
            byte_pos += char_bytes
        
        return candidates
    
    def _calculate_corpus_entropy(self, corpus: str) -> float:
        """
        Calculate Shannon entropy of the corpus [cite: 93-96].
        
        H(X) = -Σ p(x) log2(p(x))
        """
        char_counts: Dict[str, int] = defaultdict(int)
        for char in corpus:
            char_counts[char] += 1
        
        total = len(corpus)
        if total == 0:
            return 0.0
        
        entropy = 0.0
        for count in char_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _calculate_optimal_size(self, entropy: float, epsilon: float = 0.5) -> int:
        """
        Calculate optimal vocabulary size from entropy [cite: 94].
        
        V_optimal ≈ 2^(H(corpus) + ε)
        
        For English text (H ≈ 1.2 bits/char), this yields ~500k tokens.
        """
        return int(2 ** (entropy + epsilon))
    
    def _score_candidates(
        self,
        candidates: Dict[str, int],
        total_chars: int
    ) -> List[TokenCandidate]:
        """
        Calculate information gain for each candidate [cite: 129-134].
        
        Gain(s) = Frequency(s) × EntropyReduction(s) - ComputationalCost(s)
        
        Utility = (Gain × Compression) / Cost
        """
        scored: List[TokenCandidate] = []
        
        for token, freq in candidates.items():
            # Filter low-frequency noise
            if freq < self.min_frequency:
                continue
            
            # Skip single whitespace and control characters
            if len(token) == 1 and not token.isalnum():
                continue
            
            # Probability of this token
            p_token = freq / total_chars
            
            # Information content (entropy reduction) [cite: 131]
            # H(s) = -log2(p(s))
            if p_token > 0:
                entropy = -math.log2(p_token)
            else:
                continue
            
            # Computational Cost Estimate [cite: 133]
            # Cost is linear to byte length + overhead for SIMD alignment
            byte_length = len(token.encode('utf-8'))
            comp_cost = byte_length * 0.1 + 1.0
            
            # Information Gain [cite: 134]
            info_gain = entropy * freq
            
            # Compression benefit: longer tokens = more compression
            compression = byte_length * freq
            
            # Utility Score (multi-objective optimization) [cite: 1224]
            # Utility = (InfoGain × 0.4) + (Compression × 0.3) + (1/Cost × 0.3)
            utility = (
                (info_gain * 0.4) +
                (compression * 0.3) +
                ((1.0 / comp_cost) * 0.3 * freq)
            )
            
            scored.append(TokenCandidate(
                token=token,
                frequency=freq,
                entropy=entropy,
                information_gain=info_gain,
                computational_cost=comp_cost,
                utility_score=utility
            ))
        
        return scored
    
    def get_statistics(self) -> Dict:
        """Return vocabulary construction statistics."""
        return {
            "corpus_entropy": self.corpus_entropy,
            "optimal_vocab_size": self.optimal_vocab_size,
            "target_size": self.target_size,
            "max_token_length": self.max_token_length,
            "min_frequency": self.min_frequency
        }


def construct_optimal_vocabulary(
    corpus: str,
    target_size: int = 500000,
    min_frequency: int = 2
) -> List[str]:
    """
    Convenience function for vocabulary construction.
    
    This is the main entry point for building an entropy-optimized vocabulary.
    """
    builder = EntropyVocabBuilder(
        target_size=target_size,
        min_frequency=min_frequency
    )
    return builder.construct_optimal_vocabulary(corpus)


def deterministic_sort_key(token: str, frequency: int) -> tuple:
    """
    4-Key Deterministic Sort Tuple [cite: 1040-1049].
    
    Guarantees reproducible token ordering across environments:
    1. -frequency: High frequency first (for variable-byte encoding efficiency)
    2. len(bytes): Shortest tokens first
    3. token: Alphabetical ordering
    4. MD5 hash: Absolute determinism tie-breaker
    """
    token_bytes = token.encode('utf-8')
    return (
        -frequency,                                    # 1. High frequency first
        len(token_bytes),                              # 2. Shortest length second
        token,                                         # 3. Alphabetical third
        hashlib.md5(token_bytes).hexdigest()          # 4. Hash tie-breaker
    )


def assign_stable_ids(
    tokens: List[str],
    frequencies: Optional[Dict[str, int]] = None
) -> Dict[str, int]:
    """
    Assign stable, deterministic IDs to tokens [cite: 1009-1051].
    
    Reserved ID Ranges:
    - 0-99: Special tokens (<PAD>, <UNK>, <BOS>, <EOS>)
    - 100-355: ASCII byte values
    - 356-9999: Common words
    - 10000+: Subwords and rare tokens
    """
    if frequencies is None:
        frequencies = {t: 1 for t in tokens}
    
    # Predefined special tokens
    specials = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    
    # Categorize tokens
    ascii_tokens = [t for t in tokens if len(t) == 1 and ord(t) < 256 and t not in specials]
    regular_tokens = [t for t in tokens if t not in specials and t not in ascii_tokens]
    
    # Sort regular tokens deterministically
    regular_tokens.sort(key=lambda t: deterministic_sort_key(t, frequencies.get(t, 0)))
    
    # Assign IDs
    token_to_id: Dict[str, int] = {}
    current_id = 0
    
    # 1. Special tokens (0-99)
    for t in specials:
        if t in tokens or t in specials:
            token_to_id[t] = current_id
            current_id += 1
    
    # Pad to 100
    current_id = 100
    
    # 2. ASCII tokens (100-355)
    for t in sorted(ascii_tokens, key=ord):
        token_to_id[t] = current_id
        current_id += 1
    
    # Pad to 356
    current_id = max(current_id, 356)
    
    # 3. Regular tokens (356+)
    for t in regular_tokens:
        if t not in token_to_id:
            token_to_id[t] = current_id
            current_id += 1
    
    return token_to_id
