import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any
from enum import Enum

@dataclass(slots=True, frozen=True)
class TokenMetadata:
    """
    Comprehensive metadata for vocabulary tokens supporting stable ID assignment.
    
    As defined in Section 6.1 (Memory Layout Optimization) [cite: 387-393], 
    using slots reduces memory usage by 40-60%.
    """
    token: str
    frequency: int
    first_seen_hash: str
    category: str
    length_bytes: int

class TokenCategory(str, Enum):
    SPECIAL = "special_tokens"
    ASCII = "ascii_chars"
    COMMON = "common_words"
    SUBWORD = "subwords"
    RARE = "rare_tokens"

class StableVocabularyManager:
    """
    Manages token ID assignment with deterministic, reproducible behavior.
    
    Implements the logic from Section 8.1 ensuring that token IDs remain
    consistent across different environments and versions [cite: 990-993].
    """

    # Reserved ranges as defined in [cite: 1009]
    RESERVED_RANGES: Dict[str, range] = {
        TokenCategory.SPECIAL: range(0, 100),       # <PAD>, <UNK>, <BOS>, etc.
        TokenCategory.ASCII: range(100, 356),       # All byte values
        TokenCategory.COMMON: range(356, 10000),    # High-frequency words
        TokenCategory.SUBWORD: range(10000, 500000),# BPE-style subwords
        TokenCategory.RARE: range(500000, 1000000)  # Low-frequency/Specialized
    }

    def __init__(self, base_vocabulary: Optional[List[str]] = None):
        self.token_metadata: Dict[str, TokenMetadata] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        
        # Initialize with base vocabulary if provided
        if base_vocabulary:
            self._assign_base_token_ids(base_vocabulary)

    def _estimate_token_frequency(self, token: str, category: str) -> int:
        """
        Estimates frequency for initial sorting. In a real scenario, this would
        come from training data. Here we heuristicize for stability.
        """
        if category == TokenCategory.ASCII:
            return 1_000_000
        if category == TokenCategory.SPECIAL:
            return 1_000_000_000
        # Longer tokens generally less frequent in natural language (Zipf's law)
        return int(1_000_000 / (len(token) + 1))

    def _categorize_token(self, token: str) -> str:
        """Categorizes a single token into the defined ranges."""
        if token.startswith("<") and token.endswith(">"):
            return TokenCategory.SPECIAL
        if len(token.encode('utf-8')) == 1:
            return TokenCategory.ASCII
        # Heuristic: Short alphanumeric are common, complex/long are subwords/rare
        if len(token) < 6 and token.isalpha():
            return TokenCategory.COMMON
        if len(token) < 16:
            return TokenCategory.SUBWORD
        return TokenCategory.RARE

    def _deterministic_sort(self, tokens: List[str], category: str) -> List[str]:
        """
        Sort tokens deterministically using the 4-key algorithm from Section 8.1[cite: 1040].
        
        Sort Keys:
        1. Frequency (Descending)
        2. Length (Ascending/Descending based on category)
        3. Lexicographic (Ascending)
        4. MD5 Hash (Ascending for tie-breaking)
        """
        def sort_key(t: str) -> tuple:
            freq = self._estimate_token_frequency(t, category)
            length = len(t.encode('utf-8'))
            lex = t
            t_hash = hashlib.md5(t.encode('utf-8')).hexdigest()

            # For common words, we prioritize frequency
            if category in [TokenCategory.COMMON, TokenCategory.SPECIAL]:
                return (-freq, length, lex, t_hash)
            # For subwords, we prioritize systematic length ordering
            return (length, lex, -freq, t_hash)

        return sorted(tokens, key=sort_key)

    def _assign_base_token_ids(self, tokens: List[str]) -> None:
        """Assigns IDs to the initial vocabulary batch."""
        categorized: Dict[str, List[str]] = {cat: [] for cat in self.RESERVED_RANGES}
        
        for token in tokens:
            cat = self._categorize_token(token)
            categorized[cat].append(token)

        for category, token_range in self.RESERVED_RANGES.items():
            sorted_tokens = self._deterministic_sort(categorized[category], category)
            
            # Assign strictly within range
            current_id = token_range.start
            for token in sorted_tokens:
                if current_id >= token_range.stop:
                    # In production, we would log a warning or spill over carefully
                    # For strict adherence, we stop or move to RARE
                    if category != TokenCategory.RARE:
                        # Spillover logic could go here
                        pass
                    continue
                
                self._register_token(token, current_id, category)
                current_id += 1

    def _register_token(self, token: str, token_id: int, category: str) -> None:
        """Internal helper to update all mappings."""
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        self.token_metadata[token] = TokenMetadata(
            token=token,
            frequency=self._estimate_token_frequency(token, category),
            first_seen_hash=hashlib.md5(token.encode('utf-8')).hexdigest(),
            category=category,
            length_bytes=len(token.encode('utf-8'))
        )

    def add_tokens_incrementally(self, new_tokens: List[str], preserve_existing: bool = True) -> Dict[str, int]:
        """
        Adds new tokens while maintaining ID stability[cite: 1051].
        
        Returns:
            Dictionary mapping new tokens to their assigned IDs.
        """
        new_assignments = {}
        tokens_to_process = [t for t in new_tokens if t not in self.token_to_id]
        
        # Categorize
        categorized: Dict[str, List[str]] = {cat: [] for cat in self.RESERVED_RANGES}
        for token in tokens_to_process:
            cat = self._categorize_token(token)
            categorized[cat].append(token)

        # Assign
        for category, tokens in categorized.items():
            if not tokens: 
                continue
                
            token_range = self.RESERVED_RANGES[category]
            sorted_tokens = self._deterministic_sort(tokens, category)
            
            # Find holes in the current range or append to end
            used_ids_in_range = {
                id_ for id_ in self.id_to_token 
                if token_range.start <= id_ < token_range.stop
            }
            
            # Simple linear probe for next available ID
            candidate_id = token_range.start
            for token in sorted_tokens:
                while candidate_id in used_ids_in_range and candidate_id < token_range.stop:
                    candidate_id += 1
                
                if candidate_id >= token_range.stop:
                    # Fallback to RARE if range exhausted
                    if category != TokenCategory.RARE:
                        fallback_range = self.RESERVED_RANGES[TokenCategory.RARE]
                        # This recursive logic would need boundary checks in full impl
                        # For now, we assume RARE has space
                        # Simplified logic for brevity: find first open spot in RARE
                        pass 
                    continue

                self._register_token(token, candidate_id, category)
                new_assignments[token] = candidate_id
                used_ids_in_range.add(candidate_id)
        
        return new_assignments