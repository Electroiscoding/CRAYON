"""
Adaptive Vocabulary Manager Module.

Implements Section 8.2 of the XERV Crayon Engineering Treatise:
- Real-time entropy monitoring
- Adaptive vocabulary updates with feedback control
- Unknown token handling with candidate extraction
"""

import time
import math
from collections import defaultdict, deque
from typing import List, Tuple, Dict, Any, Optional, Set

from ..core.vocabulary import CrayonVocab
from .stability import StableVocabularyManager


class AdaptiveVocabularyManager:
    """
    Manages vocabulary adaptation for out-of-distribution text processing.
    
    Implements the control loop defined in Section 8.2:
    dV/dt = eta * grad_V [Performance(V,t) - Complexity(V)][cite: 140].
    
    Features:
    - Rolling window unknown token rate monitoring
    - Entropy-guided candidate extraction
    - Multi-objective utility ranking
    - Cooldown-based adaptation triggering
    """

    def __init__(self, 
                 base_vocab_manager: StableVocabularyManager,
                 core_vocab: CrayonVocab,
                 adaptation_threshold: float = 0.15,
                 min_candidate_frequency: int = 5,
                 max_candidates_per_batch: int = 50,
                 cooldown_seconds: float = 300.0):
        """
        Initialize the adaptive manager.
        
        Args:
            base_vocab_manager: Stable ID assignment manager
            core_vocab: Core vocabulary for tokenization
            adaptation_threshold: Unknown rate threshold for triggering adaptation
            min_candidate_frequency: Minimum frequency for candidate consideration
            max_candidates_per_batch: Maximum tokens to add per adaptation event
            cooldown_seconds: Minimum time between adaptations
        """
        self.vocab_manager = base_vocab_manager
        self.core_vocab = core_vocab
        self.adaptation_threshold = adaptation_threshold
        self.min_candidate_frequency = min_candidate_frequency
        self.max_candidates_per_batch = max_candidates_per_batch
        self.cooldown_seconds = cooldown_seconds
        
        # Rolling window for effectiveness monitoring [cite: 1106]
        self.unknown_token_rate: deque = deque(maxlen=1000)
        self.candidate_tokens: Dict[str, int] = defaultdict(int)
        self.candidate_lengths: Dict[str, List[int]] = defaultdict(list)
        
        # Active unknown spans for extraction
        self._current_unknown_spans: List[Tuple[int, int]] = []
        
        self.processing_stats = {
            'total_tokens': 0,
            'unknown_tokens': 0,
            'adaptation_events': 0,
            'last_adaptation_time': 0.0,
            'total_texts_processed': 0,
            'candidates_extracted': 0
        }

    def tokenize_with_adaptation(self, text: str) -> Tuple[List[int], Dict[str, Any]]:
        """
        Tokenizes text while monitoring for adaptation opportunities[cite: 1120].
        
        Returns:
            Tuple(List[int], MetadataDict with adaptation info)
        """
        # 1. Standard Tokenization
        tokens = self.core_vocab.tokenize(text)
        
        # 2. Analyze Unknowns
        unk_id = self.core_vocab.unk_token_id
        unknown_positions = [i for i, t in enumerate(tokens) if t == unk_id]
        unknown_count = len(unknown_positions)
        total = len(tokens)
        
        # 3. Update Statistics
        self.processing_stats['total_tokens'] += total
        self.processing_stats['unknown_tokens'] += unknown_count
        self.processing_stats['total_texts_processed'] += 1
        
        current_rate = unknown_count / total if total > 0 else 0.0
        self.unknown_token_rate.append(current_rate)

        # 4. Extract Candidates from unknown spans
        if unknown_count > 0:
            self._extract_candidates_from_text(text, tokens, unknown_positions)

        # 5. Trigger Adaptation? [cite: 1157]
        adaptation_metadata = {
            'unknown_rate': current_rate,
            'total_tokens': total,
            'unknown_count': unknown_count,
            'adaptation_triggered': False
        }
        
        if self._should_trigger_adaptation():
            result = self._perform_vocabulary_adaptation()
            adaptation_metadata.update(result)
            adaptation_metadata['adaptation_triggered'] = True

        return tokens, adaptation_metadata

    def _extract_candidates_from_text(
        self, 
        text: str, 
        tokens: List[int], 
        unknown_positions: List[int]
    ) -> None:
        """
        Extract candidate tokens from text regions that caused UNK tokens.
        
        Maps token positions back to character positions to identify
        untokenized spans for vocabulary expansion.
        """
        if not unknown_positions:
            return
            
        unk_id = self.core_vocab.unk_token_id
        text_len = len(text)
        
        # Reconstruct character positions from tokens
        # Each UNK corresponds to exactly 1 character in our tokenizer
        char_pos = 0
        unknown_chars: Set[int] = set()
        
        for i, token_id in enumerate(tokens):
            if token_id == unk_id:
                if char_pos < text_len:
                    unknown_chars.add(char_pos)
                char_pos += 1
            else:
                # Get token string length
                token_str = self.core_vocab.id_to_token.get(token_id, '')
                char_pos += len(token_str)
        
        # Find contiguous unknown spans
        if not unknown_chars:
            return
            
        sorted_positions = sorted(unknown_chars)
        spans: List[Tuple[int, int]] = []
        span_start = sorted_positions[0]
        span_end = span_start
        
        for pos in sorted_positions[1:]:
            if pos == span_end + 1:
                span_end = pos
            else:
                spans.append((span_start, span_end + 1))
                span_start = pos
                span_end = pos
        spans.append((span_start, span_end + 1))
        
        # Extract candidate substrings from spans with context
        for start, end in spans:
            # Extend context window for better candidates
            context_start = max(0, start - 2)
            context_end = min(text_len, end + 2)
            
            # Extract all substrings in the span (up to SIMD limit of 16 bytes)
            for length in range(1, min(17, context_end - context_start + 1)):
                for i in range(context_start, context_end - length + 1):
                    candidate = text[i:i + length]
                    
                    # Skip if already in vocabulary
                    if candidate in self.core_vocab.token_to_id:
                        continue
                    
                    # Skip control characters and whitespace-only
                    if not candidate.strip() or not candidate.isprintable():
                        continue
                    
                    # Skip if byte length exceeds SIMD limit
                    if len(candidate.encode('utf-8')) > 16:
                        continue
                    
                    self.candidate_tokens[candidate] += 1
                    self.candidate_lengths[candidate].append(length)
                    self.processing_stats['candidates_extracted'] += 1

    def _should_trigger_adaptation(self) -> bool:
        """
        Determines trigger based on threshold and cooldown[cite: 1157].
        
        Criteria:
        1. Minimum sample size (100 recent tokenizations)
        2. Unknown rate exceeds threshold
        3. Cooldown period elapsed
        4. Candidate pool has viable options
        """
        # Check minimum samples
        if len(self.unknown_token_rate) < 100:
            return False
        
        # Calculate recent unknown rate
        recent_rate = sum(self.unknown_token_rate) / len(self.unknown_token_rate)
        
        # Check threshold
        if recent_rate < self.adaptation_threshold:
            return False
            
        # Check cooldown (default 5 minutes) [cite: 1173]
        current_time = time.time()
        if current_time - self.processing_stats['last_adaptation_time'] < self.cooldown_seconds:
            return False
        
        # Check candidate pool
        viable_candidates = sum(
            1 for freq in self.candidate_tokens.values() 
            if freq >= self.min_candidate_frequency
        )
        if viable_candidates < 5:
            return False
            
        return True

    def _rank_candidates_by_utility(self) -> List[Tuple[str, float]]:
        """
        Ranks candidates using the multi-objective utility function[cite: 1224].
        
        Utility = (Compression × 0.4) + (1/Speed × 0.3) + (Coherence × 0.3)
        
        Where:
        - Compression: bits saved = len(token) × frequency
        - Speed: inverse of lookup cost (favors shorter tokens)
        - Coherence: linguistic quality score (alpha = 1.0, mixed = 0.5)
        """
        results: List[Tuple[str, float]] = []
        
        for token, freq in self.candidate_tokens.items():
            # Filter low-frequency noise
            if freq < self.min_candidate_frequency:
                continue
            
            # Already in vocabulary check
            if token in self.core_vocab.token_to_id:
                continue
            
            # Compression benefit: bytes saved per occurrence
            byte_len = len(token.encode('utf-8'))
            compression_benefit = byte_len * freq
            
            # Speed impact: shorter tokens are faster to process
            # Normalized to 0-1 range (16 bytes max)
            speed_factor = 1.0 - (byte_len / 16.0)
            
            # Coherence: linguistic quality heuristics
            coherence = 1.0
            if token.isalpha():
                coherence = 1.0  # Pure alphabetic
            elif token.isalnum():
                coherence = 0.8  # Alphanumeric
            elif any(c.isalpha() for c in token):
                coherence = 0.6  # Mixed with some letters
            else:
                coherence = 0.3  # Punctuation/symbols
            
            # Multi-objective utility [cite: 1224]
            utility = (
                (compression_benefit * 0.4) +
                (speed_factor * freq * 0.3) +
                (coherence * freq * 0.3)
            )
            
            results.append((token, utility))
            
        return sorted(results, key=lambda x: x[1], reverse=True)

    def _perform_vocabulary_adaptation(self) -> Dict[str, Any]:
        """
        Executes the vocabulary update[cite: 1179].
        
        Steps:
        1. Rank candidates by utility
        2. Select top-N candidates
        3. Add to stable vocabulary manager
        4. Clear candidate pool
        5. Update statistics
        """
        candidates = self._rank_candidates_by_utility()
        
        # Select top candidates up to batch limit
        selected = [c[0] for c in candidates[:self.max_candidates_per_batch]]
        
        if not selected:
            return {
                'new_tokens': 0,
                'candidates_considered': len(candidates),
                'timestamp': time.time()
            }
        
        # Add to vocabulary manager with stable ID assignment
        new_ids = self.vocab_manager.add_tokens_incrementally(selected)
        
        # Note: In production, would need to rebuild C-trie here
        # This requires re-calling _build_c_trie on the core vocab
        # For now, new tokens will use Python fallback until restart
        
        # Clear candidate pool after successful adaptation
        self.candidate_tokens.clear()
        self.candidate_lengths.clear()
        
        # Update statistics
        self.processing_stats['last_adaptation_time'] = time.time()
        self.processing_stats['adaptation_events'] += 1
        
        return {
            'new_tokens': len(new_ids),
            'tokens_added': list(new_ids.keys()),
            'candidates_considered': len(candidates),
            'timestamp': time.time()
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Return current processing and adaptation statistics."""
        avg_unknown_rate = (
            sum(self.unknown_token_rate) / len(self.unknown_token_rate)
            if self.unknown_token_rate else 0.0
        )
        
        return {
            **self.processing_stats,
            'current_unknown_rate': avg_unknown_rate,
            'candidate_pool_size': len(self.candidate_tokens),
            'viable_candidates': sum(
                1 for f in self.candidate_tokens.values() 
                if f >= self.min_candidate_frequency
            )
        }

    def force_adaptation(self) -> Dict[str, Any]:
        """Force an immediate adaptation regardless of thresholds."""
        return self._perform_vocabulary_adaptation()

    def clear_candidates(self) -> None:
        """Clear the candidate token pool."""
        self.candidate_tokens.clear()
        self.candidate_lengths.clear()
        self.processing_stats['candidates_extracted'] = 0