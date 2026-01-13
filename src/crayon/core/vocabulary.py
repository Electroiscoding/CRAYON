"""
Memory-optimized Vocabulary with O(1) lookup and O(L) longest-match.

Implements the hybrid Trie/Hash map data structure described in Section 4.2
of the XERV Crayon Engineering Treatise.

Features:
- Entropy-guided vocabulary construction from corpus
- Hardware-aligned Trie for SIMD acceleration
- Stable, deterministic ID assignment
- Cache-optimized memory layout
- Batteries-included default vocabulary builder
"""

from typing import List, Dict, Tuple, Optional, Any, Iterator
import sys


class CrayonVocab:
    """
    Memory-optimized vocabulary with O(1) lookup and O(L) longest-match.
    
    Automatically upgrades to AVX2 C-Backend if available [cite: 178-190].
    
    Data Structures:
    - Trie: Optimized for prefix matching and cache locality
    - Hash Table: Reverse mapping for decoding
    - C-Extension: SIMD-accelerated trie for production throughput
    """

    def __init__(self, tokens: List[str], unk_token: str = "<UNK>"):
        """
        Initialize vocabulary from pre-computed token list.
        
        For building vocabulary from corpus, use:
        - CrayonVocab.from_corpus() for custom corpus
        - CrayonVocab.from_default_sources() for batteries-included
        
        Args:
            tokens: List of token strings (order determines IDs)
            unk_token: Unknown token representation
        """
        self.size = len(tokens)
        self.unk_token = unk_token
        
        # 1. Standard Python mappings (for fallback/decoding)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(tokens)}
        self.unk_token_id = self.token_to_id.get(unk_token, 0)
        
        # 2. Build Python Trie (Fallback)
        self._root: Dict[str, Any] = {'children': {}, 'token_id': -1}
        self._build_python_trie(tokens)
        
        # 3. Build C-Extension Trie (Production Path)
        self._c_trie: Optional[Any] = None
        self._c_ext_available = False
        self._build_c_trie(tokens)

    @classmethod
    def from_corpus(
        cls,
        corpus: str,
        target_size: int = 500000,
        min_frequency: int = 2,
        unk_token: str = "<UNK>"
    ) -> "CrayonVocab":
        """
        Build vocabulary from corpus using entropy-guided construction.
        
        Implements Algorithm 3.1 [cite: 126-135]:
        - Extract substring candidates up to SIMD limit (16 bytes)
        - Calculate information gain with entropy reduction
        - Select top-K candidates maximizing utility score
        
        Args:
            corpus: Training text for vocabulary construction
            target_size: Maximum vocabulary size (default 500k)
            min_frequency: Minimum token frequency threshold
            unk_token: Unknown token representation
            
        Returns:
            CrayonVocab instance with optimized vocabulary
        """
        from .vocab_builder import EntropyVocabBuilder
        
        builder = EntropyVocabBuilder(
            target_size=target_size,
            min_frequency=min_frequency,
            special_tokens=["<PAD>", unk_token, "<BOS>", "<EOS>"]
        )
        tokens = builder.construct_optimal_vocabulary(corpus)
        
        return cls._stabilize_and_create(tokens, unk_token)

    @classmethod
    def from_corpus_iterator(
        cls,
        corpus_iterator: Iterator[str],
        vocab_size: int = 500000,
        unk_token: str = "<UNK>"
    ) -> "CrayonVocab":
        """
        Build vocabulary from streaming corpus iterator.
        
        Args:
            corpus_iterator: Iterator yielding text chunks
            vocab_size: Maximum vocabulary size
            unk_token: Unknown token representation
            
        Returns:
            CrayonVocab instance with optimized vocabulary
        """
        from crayon.training import train_vocabulary
        
        raw_tokens = train_vocabulary(corpus_iterator, target_size=vocab_size)
        return cls._stabilize_and_create(raw_tokens, unk_token)

    @classmethod
    def from_default_sources(
        cls,
        vocab_size: int = 500000,
        unk_token: str = "<UNK>",
        progress_callback: Optional[callable] = None
    ) -> "CrayonVocab":
        """
        [Batteries Included] Build high-entropy vocabulary from curated sources.
        
        Streams data directly from:
        - Xerv-AI/GRAD (Graduate Mathematics)
        - Xerv-AI/Physics-dataset-700 (Scientific Reasoning)
        - Xerv-AI/RainDrop-DTS (General Instruction Following)
        - Tiny Shakespeare (Classical Literature)
        - Built-in corpus (Baseline Coverage)
        
        No local files required - data streams directly into the entropy engine.
        
        Args:
            vocab_size: Maximum vocabulary size (default 500k)
            unk_token: Unknown token representation
            progress_callback: Optional callback for progress updates
            
        Returns:
            CrayonVocab instance with production-grade vocabulary
            
        Example:
            >>> vocab = CrayonVocab.from_default_sources(vocab_size=50000)
            >>> tokens = vocab.tokenize("Hello, world!")
        """
        from ..training import build_default_vocabulary
        
        raw_tokens = build_default_vocabulary(
            target_size=vocab_size,
            progress_callback=progress_callback
        )
        return cls._stabilize_and_create(raw_tokens, unk_token)

    @classmethod
    def from_file(cls, vocab_path: str, unk_token: str = "<UNK>") -> "CrayonVocab":
        """
        Load vocabulary from file (one token per line).
        
        Args:
            vocab_path: Path to vocabulary file
            unk_token: Unknown token representation
            
        Returns:
            CrayonVocab instance
        """
        with open(vocab_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
        return cls(tokens, unk_token=unk_token)

    @classmethod
    def from_json(cls, json_path: str, unk_token: str = "<UNK>") -> "CrayonVocab":
        """
        Load vocabulary from JSON file.
        
        Supports formats:
        - List of tokens: ["token1", "token2", ...]
        - Dict mapping: {"token1": 0, "token2": 1, ...}
        
        Args:
            json_path: Path to JSON vocabulary file
            unk_token: Unknown token representation
            
        Returns:
            CrayonVocab instance
        """
        import json
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            tokens = data
        elif isinstance(data, dict):
            # Sort by ID to maintain order
            tokens = [k for k, v in sorted(data.items(), key=lambda x: x[1])]
        else:
            raise ValueError(f"Unsupported JSON format: expected list or dict")
        
        return cls(tokens, unk_token=unk_token)

    @classmethod
    def _stabilize_and_create(
        cls, 
        raw_tokens: List[str],
        unk_token: str = "<UNK>"
    ) -> "CrayonVocab":
        """
        Internal helper to apply Stable ID sorting before creation.
        
        Uses StableVocabularyManager for deterministic ID assignment.
        """
        from ..adaptive.stability import StableVocabularyManager
        
        stable_mgr = StableVocabularyManager()
        stable_mgr.add_tokens_incrementally(raw_tokens, preserve_existing=False)
        
        # Extract tokens in ID order
        max_id = max(stable_mgr.id_to_token.keys()) if stable_mgr.id_to_token else -1
        sorted_tokens: List[Optional[str]] = [None] * (max_id + 1)
        
        for tid, token in stable_mgr.id_to_token.items():
            if 0 <= tid < len(sorted_tokens):
                sorted_tokens[tid] = token
        
        # Filter None values and create
        final_tokens = [t for t in sorted_tokens if t is not None]
        return cls(final_tokens, unk_token=unk_token)

    def _build_python_trie(self, tokens: List[str]) -> None:
        """Constructs pure Python trie structure for fallback."""
        for i, token in enumerate(tokens):
            node = self._root
            for char in token:
                if char not in node['children']:
                    node['children'][char] = {'children': {}, 'token_id': -1}
                node = node['children'][char]
            node['token_id'] = i

    def _build_c_trie(self, tokens: List[str]) -> None:
        """
        Attempts to build the SIMD-optimized C Trie [cite: 402-412].
        
        Falls back to Python trie if C extension unavailable.
        """
        try:
            from ..c_ext import _core
            # Call the C build_trie function
            self._c_trie = _core.build_trie(tokens)
            self._c_ext_available = True
        except ImportError:
            # C extension not compiled
            self._c_ext_available = False
        except Exception as e:
            # Build failed for some reason
            print(
                f"[Crayon Warning] Failed to build C-Trie: {e}. Using Python fallback.", 
                file=sys.stderr
            )
            self._c_ext_available = False

    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text to token IDs.
        
        Uses C-extension with SIMD acceleration if available,
        otherwise falls back to pure Python implementation.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of token IDs
        """
        from .tokenizer import crayon_tokenize
        return crayon_tokenize(text, self)
    
    def longest_match(
        self, 
        text: str, 
        position: int, 
        max_lookahead: int = 256
    ) -> Tuple[int, int]:
        """
        Python implementation of longest match (Fallback).
        
        Optimizations [cite: 193-196]:
        - Early termination on mismatch
        - Limited lookahead for bounded complexity
        
        Args:
            text: Input text
            position: Starting position
            max_lookahead: Maximum characters to look ahead
            
        Returns: 
            Tuple of (token_id, match_length)
        """
        node = self._root
        best_match_length = 0
        best_token_id = -1
        current_length = 0
        
        end_pos = min(position + max_lookahead, len(text))
        
        for i in range(position, end_pos):
            char = text[i]
            if char not in node['children']:
                break
            node = node['children'][char]
            current_length += 1
            if node['token_id'] != -1:
                best_match_length = current_length
                best_token_id = node['token_id']
                
        return best_token_id, best_match_length
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to string.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded string
        """
        return ''.join(self.id_to_token.get(tid, self.unk_token) for tid in token_ids)
    
    def encode(self, text: str) -> List[int]:
        """
        Alias for tokenize() for API compatibility.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        return self.tokenize(text)
    
    def save(self, path: str, format: str = "txt") -> None:
        """
        Save vocabulary to file.
        
        Args:
            path: Output file path
            format: Output format ("txt" or "json")
        """
        if format == "txt":
            with open(path, 'w', encoding='utf-8') as f:
                for i in range(self.size):
                    f.write(self.id_to_token[i] + '\n')
        elif format == "json":
            import json
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.token_to_id, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the token to ID mapping."""
        return self.token_to_id.copy()
    
    def __len__(self) -> int:
        return self.size
    
    def __contains__(self, token: str) -> bool:
        return token in self.token_to_id
    
    def __getitem__(self, key):
        """
        Get token ID by token string, or token string by ID.
        
        Args:
            key: Token string or token ID
            
        Returns:
            Token ID if key is string, token string if key is int
        """
        if isinstance(key, str):
            return self.token_to_id.get(key, self.unk_token_id)
        elif isinstance(key, int):
            return self.id_to_token.get(key, self.unk_token)
        raise TypeError(f"Key must be str or int, not {type(key)}")
    
    def __repr__(self) -> str:
        return (
            f"CrayonVocab(size={self.size}, "
            f"c_ext={'enabled' if self._c_ext_available else 'disabled'})"
        )