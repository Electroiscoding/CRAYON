
"""
Double-Array Trie (DAT) Compiler for Crayon.
Compiles a sorted vocabulary list into a highly compressed, cache-local binary format (.dat).

Algorithm:
- Base[s] + c = t
- Check[t] = s
"""

import struct
import sys
import array
from typing import List, Tuple, Dict

class DATBuilder:
    def __init__(self):
        # Arrays: base and check. 
        # Initial size estimate: 2x vocab size * avg length is usually overkill but safe.
        # We will resize dynamically.
        self.base = array.array('i', [0] * 1024)
        self.check = array.array('i', [0] * 1024)
        self.used = array.array('b', [0] * 1024) # Bitset for allocation
        self.check[0] = 0 # Root check is typically 0
        self.size = 1024
        self.max_idx = 0
        
        # Token ID mapping
        self.output = {} # state_index -> token_id

    def _resize(self, new_size):
        if new_size <= self.size:
            return
        # Python arrays scale efficiently
        extension = [0] * (new_size - self.size)
        self.base.extend(extension)
        self.check.extend(extension)
        self.used.extend([0] * (new_size - self.size))
        self.size = new_size

    def _find_base(self, children_keys: List[int]) -> int:
        """Finds a base offset 'b' such that check[b + c] are all empty for each c in children."""
        if not children_keys:
            return 1 # Leaf
            
        first = children_keys[0]
        # Start searching from 1
        b = 1 
        while True:
            # First candidate check: base + first_child
            pos = b + first
            if pos >= self.size:
                self._resize(pos + 256)
                
            if self.check[pos] != 0:
                # Collision for first child, move forward
                b += 1
                continue
            
            # Now verify all other children
            overlap = False
            max_pos = 0
            for k in children_keys:
                p = b + k
                if p >= self.size:
                    self._resize(p + 256)
                max_pos = max(max_pos, p)
                
                if self.check[p] != 0:
                    overlap = True
                    break
            
            if not overlap:
                return b
            
            b += 1

    def build(self, tokens: List[str]) -> bytes:
        """
        Builds the Double-Array Trie from sorted tokens.
        """
        # 1. Build Standard Trie first (Intermediate representation)
        # Dictionary of node -> {char: next_node}
        trie = {'id': -1, 'children': {}}
        
        for i, token in enumerate(tokens):
            node = trie
            for char in token:
                key = ord(char)
                if key not in node['children']:
                    node['children'][key] = {'id': -1, 'children': {}}
                node = node['children'][key]
            node['id'] = i
            
        # 2. Convert to Double-Array via BFS
        # Queue: (trie_node, dat_state_index)
        queue: List[Tuple[Dict, int]] = [(trie, 0)] # Root is state 0
        
        # Mark root as used
        self.base[0] = 1
        self._resize(256) # Ensure capacity
        
        processed_count = 0
        
        while queue:
            node, state = queue.pop(0)
            
            if node['id'] != -1:
                self.output[state] = node['id']
                # Mark as terminal in base array? 
                # Technique: We usually store leaf status by negative base or separate array.
                # For Crayon, we want fast token ID retrieval.
                # We will store token_id mapping separately OR encode it.
                # Let's encode token_id as negative base: base[s] = -token_id - 1
                # BUT a node can be both transit and terminal (e.g., "apple", "apples").
                # Standard DAT handles this by specific termination char '\0' or separate array.
                # To keep it compact: We will use a separate output structure for now 
                # OR stick to the Crayon specialized TrieNode structure.
                
                # Solution: We will store token_ids in a separate array `terminals` which parallels check/base.
                # If terminals[s] != -1, it's a match.
                pass

            children = node['children']
            if not children:
                continue
                
            sorted_keys = sorted(children.keys())
            
            # Find a valid base for this state
            base_offset = self._find_base(sorted_keys)
            self.base[state] = base_offset
            
            # set check and prepare children
            for k in sorted_keys:
                next_state = base_offset + k
                self.check[next_state] = state
                self.used[next_state] = 1 # Mark
                self.max_idx = max(self.max_idx, next_state)
                
                queue.append((children[k], next_state))
                
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"Compiled {processed_count} states...", end='\r')

        print(f"\nDAT Construction Complete. {self.max_idx} states.")
        return self._serialize()

    def _serialize(self) -> bytes:
        """
        Format:
        [HEADER: 16 bytes]
          - Magic: "CRYN" (4)
          - Version: 1 (4)
          - Size: int (4)
        [BODY]
          - Base: int32 * size
          - Check: int32 * size
          - Terminals: int32 * size (Token mapping)
        """
        # Optimize size
        final_size = self.max_idx + 1
        
        # Build terminals array
        terminals = array.array('i', [-1] * final_size)
        for state, pid in self.output.items():
            if state < final_size:
                terminals[state] = pid
                
        header = struct.pack('<4sII', b'CRYN', 1, final_size)
        
        # Slice correct size
        final_base = self.base[:final_size]
        final_check = self.check[:final_size]
        
        print(f"Serialized Size: {(final_size * 12 + 12) / 1024 / 1024:.2f} MB")
        
        return (
            header + 
            final_base.tobytes() + 
            final_check.tobytes() + 
            terminals.tobytes()
        )

def compile_dat(tokens: List[str], output_path: str):
    builder = DATBuilder()
    data = builder.build(tokens)
    with open(output_path, 'wb') as f:
        f.write(data)
    print(f"Saved: {output_path}")

