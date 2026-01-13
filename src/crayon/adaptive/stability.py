"""
Stable Vocabulary Management Module.

Implements Section 8.1 of the XERV Crayon Engineering Treatise:
- Deterministic 4-key sorting for reproducible ID assignment
- Reserved ID ranges for token categories
- Incremental token addition with stability guarantees
"""

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum


@dataclass(slots=True, frozen=True)
class TokenMetadata:
    """
    Comprehensive metadata for vocabulary tokens.
    
    Uses slots for 40-60% memory reduction [cite: 387-393].
    """
    token: str
    frequency: int
    first_seen_hash: str
    category: str
    length_bytes: int


class TokenCategory(str, Enum):
    """Token category for ID range assignment [cite: 1009-1012]."""
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
    
    Features:
    - 4-key deterministic sort (frequency, length, lexicographic, MD5)
    - Reserved ID ranges for token categories
    - Incremental addition with stability guarantees
    """

    # Reserved ranges [cite: 1009-1012]
    RESERVED_RANGES: Dict[TokenCategory, range] = {
        TokenCategory.SPECIAL: range(0, 100),        # <PAD>, <UNK>, <BOS>, etc.
        TokenCategory.ASCII: range(100, 356),        # All printable ASCII
        TokenCategory.COMMON: range(356, 10000),     # High-frequency words
        TokenCategory.SUBWORD: range(10000, 500000), # BPE-style subwords
        TokenCategory.RARE: range(500000, 1000000)   # Low-frequency/Specialized
    }

    def __init__(self, base_vocabulary: Optional[List[str]] = None):
        self.token_metadata: Dict[str, TokenMetadata] = {}
        self.id_to_token: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self._frequency_cache: Dict[str, int] = {}
        
        if base_vocabulary:
            self._assign_base_token_ids(base_vocabulary)

    def _deterministic_sort_key(self, token: str) -> tuple:
        """
        4-Key Deterministic Sort [cite: 1040-1049].
        
        Sort Keys:
        1. -Frequency (Descending) - Common tokens get lower IDs
        2. Length (Ascending) - Shorter tokens first
        3. Lexicographic (Ascending) - Alphabetical for reproducibility
        4. MD5 Hash (Ascending) - Absolute determinism tie-breaker
        """
        freq = self._frequency_cache.get(token, 0)
        token_bytes = token.encode('utf-8')
        return (
            -freq,
            len(token_bytes),
            token,
            hashlib.md5(token_bytes).hexdigest()
        )

    def _estimate_token_frequency(self, token: str, category: TokenCategory) -> int:
        """Estimate frequency for initial sorting based on heuristics."""
        if category == TokenCategory.SPECIAL:
            return 1_000_000_000
        if category == TokenCategory.ASCII:
            return 1_000_000
        # Zipf's law: frequency inversely proportional to length
        return int(1_000_000 / (len(token) + 1))

    def _categorize_token(self, token: str) -> TokenCategory:
        """Categorize token into reserved range [cite: 1009-1012]."""
        if token.startswith("<") and token.endswith(">"):
            return TokenCategory.SPECIAL
        if len(token.encode('utf-8')) == 1 and ord(token[0]) < 256:
            return TokenCategory.ASCII
        if len(token) < 6 and token.isalpha():
            return TokenCategory.COMMON
        if len(token) < 16:
            return TokenCategory.SUBWORD
        return TokenCategory.RARE

    def _assign_base_token_ids(self, tokens: List[str]) -> None:
        """Assigns IDs to the initial vocabulary batch."""
        # Categorize all tokens
        categorized: Dict[TokenCategory, List[str]] = {
            cat: [] for cat in TokenCategory
        }
        
        for token in tokens:
            cat = self._categorize_token(token)
            categorized[cat].append(token)
            self._frequency_cache[token] = self._estimate_token_frequency(token, cat)

        # Assign IDs within each category range
        for category in TokenCategory:
            token_range = self.RESERVED_RANGES[category]
            category_tokens = categorized[category]
            
            # Sort deterministically
            sorted_tokens = sorted(category_tokens, key=self._deterministic_sort_key)
            
            current_id = token_range.start
            for token in sorted_tokens:
                if current_id >= token_range.stop:
                    # Overflow to RARE category
                    if category != TokenCategory.RARE:
                        rare_range = self.RESERVED_RANGES[TokenCategory.RARE]
                        current_id = self._find_next_available(rare_range)
                        if current_id is None:
                            continue  # Skip if no space
                    else:
                        continue
                
                self._register_token(token, current_id, category)
                current_id += 1

    def _find_next_available(self, id_range: range) -> Optional[int]:
        """Find next available ID in range."""
        for id_ in id_range:
            if id_ not in self.id_to_token:
                return id_
        return None

    def _register_token(self, token: str, token_id: int, category: TokenCategory) -> None:
        """Register token with all mappings."""
        self.token_to_id[token] = token_id
        self.id_to_token[token_id] = token
        
        freq = self._frequency_cache.get(token, 0)
        self.token_metadata[token] = TokenMetadata(
            token=token,
            frequency=freq,
            first_seen_hash=hashlib.md5(token.encode('utf-8')).hexdigest(),
            category=category.value,
            length_bytes=len(token.encode('utf-8'))
        )

    def add_tokens_incrementally(
        self,
        new_tokens: List[str],
        frequencies: Optional[Dict[str, int]] = None,
        preserve_existing: bool = True
    ) -> Dict[str, int]:
        """
        Add new tokens while maintaining ID stability [cite: 1051].
        
        Returns:
            Dictionary mapping new tokens to their assigned IDs.
        """
        if frequencies:
            self._frequency_cache.update(frequencies)
        
        new_assignments: Dict[str, int] = {}
        tokens_to_process = [t for t in new_tokens if t not in self.token_to_id]
        
        # Categorize new tokens
        categorized: Dict[TokenCategory, List[str]] = {
            cat: [] for cat in TokenCategory
        }
        for token in tokens_to_process:
            cat = self._categorize_token(token)
            categorized[cat].append(token)
            if token not in self._frequency_cache:
                self._frequency_cache[token] = self._estimate_token_frequency(token, cat)

        # Assign IDs
        for category in TokenCategory:
            tokens = categorized[category]
            if not tokens:
                continue
                
            token_range = self.RESERVED_RANGES[category]
            sorted_tokens = sorted(tokens, key=self._deterministic_sort_key)
            
            # Find available IDs in range
            used_ids = {
                id_ for id_ in self.id_to_token
                if token_range.start <= id_ < token_range.stop
            }
            
            for token in sorted_tokens:
                # Find first available slot
                candidate_id = None
                for id_ in token_range:
                    if id_ not in used_ids:
                        candidate_id = id_
                        break
                
                if candidate_id is None:
                    # Try RARE range as fallback
                    if category != TokenCategory.RARE:
                        rare_range = self.RESERVED_RANGES[TokenCategory.RARE]
                        candidate_id = self._find_next_available(rare_range)
                
                if candidate_id is not None:
                    self._register_token(token, candidate_id, category)
                    new_assignments[token] = candidate_id
                    used_ids.add(candidate_id)
        
        return new_assignments

    def get_token_metadata(self, token: str) -> Optional[TokenMetadata]:
        """Get metadata for a token."""
        return self.token_metadata.get(token)

    def export_vocabulary(self) -> List[Tuple[str, int]]:
        """Export vocabulary as sorted list of (token, id) pairs."""
        return sorted(self.token_to_id.items(), key=lambda x: x[1])
    
    def __len__(self) -> int:
        return len(self.token_to_id)
    
    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id