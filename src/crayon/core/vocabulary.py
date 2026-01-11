from typing import List, Dict, Tuple, Optional, Any

class CrayonVocab:
    """
    Memory-optimized vocabulary with O(1) lookup and O(L) longest-match.
    
    Data Structures (Section 4.2) [cite: 178-190]:
    - Trie: Optimized for prefix matching and cache locality.
    - Hash Table: Reverse mapping for decoding.
    """

    def __init__(self, tokens: List[str], unk_token: str = "<UNK>"):
        self.size = len(tokens)
        self.unk_token = unk_token
        
        # Build mappings
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(tokens)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(tokens)}
        
        self.unk_token_id = self.token_to_id.get(unk_token, 0)
        
        # Build pure Python Trie (for fallback)
        self._root = {'children': {}, 'token_id': -1}
        self._build_trie(tokens)
        
        # C-Extension hook
        self._c_trie: Optional[Any] = None 
        # In a real scenario, we would call `crayon_core.build_trie(tokens)` here

    def _build_trie(self, tokens: List[str]) -> None:
        """Constructs the trie structure."""
        for i, token in enumerate(tokens):
            node = self._root
            for char in token:
                if char not in node['children']:
                    node['children'][char] = {'children': {}, 'token_id': -1}
                node = node['children'][char]
            node['token_id'] = i

    def longest_match(self, text: str, position: int, max_lookahead: int = 16) -> Tuple[int, int]:
        """
        Find Longest matching token starting at position.
        
        Optimizations [cite: 193-196]:
        - Early termination
        - Limited lookahead
        
        Returns: (token_id, match_length)
        """
        node = self._root
        best_match_length = 0
        best_token_id = -1
        current_length = 0
        
        # Bounds checking
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

    def tokenize(self, text: str) -> List[int]:
        """Delegates to the optimized tokenizer function."""
        from .tokenizer import crayon_tokenize
        return crayon_tokenize(text, self)