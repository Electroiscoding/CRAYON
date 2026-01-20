
"""
Hyper-Production Double-Array Trie (DAT) Compiler.
Compiles standard JSON vocabulary into cache-optimized binary arrays.
Algorithm: First-Fit Linear Scan with Collision Resolution.
"""

import struct
import json
import logging
from typing import List, Dict, Tuple, Optional

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - [DAT-BUILDER] - %(message)s')

class DATBuilder:
    def __init__(self):
        # Initial size: 65536 to prevent frequent resizing
        self.init_size = 65536
        self.base = [1] * self.init_size     # Base array (Offsets)
        self.check = [-1] * self.init_size   # Check array (Parent validation)
        self.values = [-1] * self.init_size  # Value array (Token IDs)
        
        # Root node is always at index 0
        self.base[0] = 1
        self.check[0] = 0
        
        self.size = self.init_size
        self.next_check_pos = 1  # Optimization cursor

    def _resize(self, required_index: int):
        """Exponential resizing strategy to amortize cost."""
        if required_index < self.size:
            return

        new_size = max(required_index + 1024, self.size * 2)
        expand_count = new_size - self.size
        
        self.base.extend([1] * expand_count)
        self.check.extend([-1] * expand_count)
        self.values.extend([-1] * expand_count)
        self.size = new_size

    def _find_base(self, children_codes: List[int]) -> int:
        """
        Finds a base offset 'q' such that for all char_code 'c':
        check[q + c] is available (== -1).
        """
        if not children_codes:
            return 1

        # Start searching from the last known free position
        q = self.next_check_pos
        first_char = children_codes[0]

        while True:
            # Ensure we have space for the first child
            if q + first_char >= self.size:
                self._resize(q + first_char + 256)
                
            # Quick Check: Is the slot for the first child taken?
            if self.check[q + first_char] != -1:
                q += 1
                continue
            
            # Full Check: Do ALL children fit?
            collision = False
            max_idx_needed = 0
            
            for c in children_codes:
                idx = q + c
                if idx >= self.size:
                    self._resize(idx + 1024)
                
                if self.check[idx] != -1:
                    collision = True
                    break
                
                if idx > max_idx_needed:
                    max_idx_needed = idx
            
            if not collision:
                # Update optimization cursor only if we used the generic start
                if q == self.next_check_pos:
                    self.next_check_pos += 1
                return q
            
            q += 1

    def build(self, vocab: List[str]) -> None:
        """
        Compiles the list of strings into the DAT structure.
        """
        logging.info(f"Compiling vocabulary of {len(vocab)} tokens...")
        
        # Step 1: Build temporary Python Trie (Tree)
        root = {'children': {}, 'val': -1}
        for token_id, token in enumerate(vocab):
            node = root
            # Convert to bytes for raw speed processing
            for byte_val in token.encode('utf-8'):
                if byte_val not in node['children']:
                    node['children'][byte_val] = {'children': {}, 'val': -1}
                node = node['children'][byte_val]
            node['val'] = token_id

        # Step 2: BFS Traversal to Pack into Arrays
        # Queue tuple: (trie_node_dict, dat_node_index)
        queue = [(root, 0)]
        
        processed_nodes = 0
        
        while queue:
            curr_node, curr_dat_idx = queue.pop(0)
            children_map = curr_node['children']
            
            if not children_map:
                continue

            # Sort children by byte value (essential for deterministic build)
            children_bytes = sorted(children_map.keys())
            
            # Find valid base
            base_offset = self._find_base(children_bytes)
            self.base[curr_dat_idx] = base_offset
            
            # Register children in the array
            for byte_val in children_bytes:
                child_node = children_map[byte_val]
                next_dat_idx = base_offset + byte_val
                
                self.check[next_dat_idx] = curr_dat_idx
                self.values[next_dat_idx] = child_node['val']
                
                queue.append((child_node, next_dat_idx))
            
            processed_nodes += 1
                
        # Shrink arrays to actual used size to save disk space
        # Find last non-default entry
        last_used = 0
        for i in range(self.size - 1, -1, -1):
            if self.check[i] != -1 or self.base[i] != 1:
                last_used = i
                break
        
        final_size = last_used + 1
        self.base = self.base[:final_size]
        self.check = self.check[:final_size]
        self.values = self.values[:final_size]
        self.size = final_size
        
        logging.info(f"Compilation Complete. Final Array Size: {self.size}")

    def save(self, output_path: str):
        """
        Saves the memory-mappable binary format.
        Format: [MAGIC 4b][VER 4b][SIZE 4b][BASE int32 array][CHECK int32 array][VALS int32 array]
        """
        logging.info(f"Saving binary to {output_path}...")
        
        with open(output_path, "wb") as f:
            # Header
            f.write(b"CRAY") # Magic
            f.write(struct.pack("<I", 2)) # Version 2.0
            f.write(struct.pack("<I", self.size)) # Array Size
            
            # Data Arrays (Packed C Integers)
            # Use 'i' for signed 32-bit int
            fmt = f"<{self.size}i"
            f.write(struct.pack(fmt, *self.base))
            f.write(struct.pack(fmt, *self.check))
            f.write(struct.pack(fmt, *self.values))
            
        logging.info("Save successful.")
