
import mmap
import os
import json
from typing import List, Optional, Iterator, Dict, Tuple, Any
from pathlib import Path

# Try Loading Optimized Backend
try:
    from ..c_ext import crayon_fast
    _C_BACKEND_AVAILABLE = True
except ImportError:
    _C_BACKEND_AVAILABLE = False
    print("[CRAYON] Warning: AVX2 backend missing. Falling back to slow mode.")

class CrayonVocab:
    def __init__(self):
        self._mmap = None
        self.fast_mode = False
        self.unk_token_id = 1 # Spec hardcodes fallback to 1, keeping consistent.
        
        # Fallback dicts
        self.token_to_id = {}
        self.id_to_token = {}

    @classmethod
    def load_profile(cls, name: str) -> 'CrayonVocab':
        """
        Loads a profile (e.g., 'science'). 
        Checks for .dat binary first, then .json.
        """
        from .profiles import PROFILES
        if name not in PROFILES:
             raise ValueError(f"Profile {name} unknown.")

        cache_dir = Path.home() / ".cache" / "xerv" / "crayon" / "profiles"
        dat_path = cache_dir / f"vocab_{name}.dat"
        json_path = cache_dir / f"vocab_{name}.json"
        
        # If JSON doesn't exist, we might need to build it (using old logic or just fail per V2 spec)
        # Assuming build_all_profiles runs separately.
        
        vocab = cls()
        
        # 1. Try Loading Binary (Fast Path)
        if dat_path.exists() and _C_BACKEND_AVAILABLE:
            vocab._load_binary_dat(dat_path)
            # Load Python mappings lazily if needed, but for "Hyper-Fast" we might skip it 
            # OR we load them for decoding support.
            # Spec step 4 suggests pure V2 structure.
            if json_path.exists():
                 vocab._load_json_mappings(json_path) # For decoding
        # 2. Try Loading JSON (Slow Path)
        elif json_path.exists():
            print(f"[Crayon] DAT not found or Engine missing. Loading JSON {name}...")
            vocab._load_json_legacy(json_path)
            # Auto-Compile Upgrade?
            # if _C_BACKEND_AVAILABLE: 
            #    vocab._compile_and_reload(name, json_path, dat_path)
        else:
            # Trigger build if missing?
            # Keeping it simple per spec.
            print(f"Profile {name} not found at {json_path}")
            # raise FileNotFoundError(f"Profile {name} not found.")
            # Fallback to hydration if resources.py exists
            from ..resources import build_and_cache_profile
            build_and_cache_profile(name)
            if dat_path.exists():
                 vocab._load_binary_dat(dat_path)
            else:
                 vocab._load_json_legacy(json_path)
            
        return vocab

    def _load_binary_dat(self, path: Path):
        """Zero-Copy Load via mmap."""
        self.file_handle = open(path, "rb")
        # Map file to memory
        self._mmap = mmap.mmap(self.file_handle.fileno(), 0, access=mmap.ACCESS_READ)
        # Initialize C++ engine
        size = crayon_fast.load_dat(self._mmap)
        self.fast_mode = True
        # print(f"[CRAYON] Loaded AVX2 Engine. Size: {size}")

    def _load_json_legacy(self, path: Path):
        """Legacy slow loader."""
        with open(path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
        
        if isinstance(tokens, list):
             data = tokens
        elif isinstance(tokens, dict):
             # Sort by ID
             data = [k for k, v in sorted(tokens.items(), key=lambda x: x[1])]
             
        self.token_to_id = {t: i for i, t in enumerate(data)}
        self.id_to_token = {i: t for i, t in enumerate(data)}
        self.fast_mode = False

    def _load_json_mappings(self, path: Path):
         """Load just the mappings for decoding support."""
         with open(path, 'r', encoding='utf-8') as f:
            tokens = json.load(f)
         if isinstance(tokens, list):
             data = tokens
         else:
             data = [k for k, v in sorted(tokens.items(), key=lambda x: x[1])]
         self.token_to_id = {t: i for i, t in enumerate(data)}
         self.id_to_token = {i: t for i, t in enumerate(data)}

    def tokenize(self, text: str) -> List[int]:
        if self.fast_mode:
            # CALL C++ DIRECTLY
            return crayon_fast.tokenize(text)
        else:
            # SLOW PYTHON FALLBACK
            return self._python_tokenize(text)

    def _python_tokenize(self, text: str) -> List[int]:
        # Simple longest match logic for fallback
        tokens = []
        pos = 0
        n = len(text)
        while pos < n:
            match = False
            # Check decreasing lengths (naive)
            for l in range(min(20, n - pos), 0, -1):
                sub = text[pos:pos+l]
                if sub in self.token_to_id:
                    tokens.append(self.token_to_id[sub])
                    pos += l
                    match = True
                    break
            if not match:
                tokens.append(1) # UNK
                pos += 1
        return tokens
    
    # Keeping minimal API compatibility
    def __len__(self):
        return len(self.token_to_id) if self.token_to_id else 0