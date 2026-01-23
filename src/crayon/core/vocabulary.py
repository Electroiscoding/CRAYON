
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
        Checks strictly in this order:
        1. Package installation directory (fastest, standard for pip install)
        2. User cache directory (dev/legacy)
        3. Fallback: Build on demand (slow)
        """
        from .profiles import PROFILES
        if name not in PROFILES:
             raise ValueError(f"Profile {name} unknown.")

        # PATH 1: Package Directory (Pre-bundled)
        # This is where 'pip install' puts the files
        try:
            import importlib.resources
            # Modern python resource access
            pkg_dat_path = Path(importlib.resources.files('crayon.resources.dat') / f"vocab_{name}.dat")
            pkg_json_path = Path(importlib.resources.files('crayon.resources.dat') / f"vocab_{name}.json")
        except (ImportError, TypeError):
            # Fallback for older python or non-standard installs
            pkg_dir = Path(__file__).parent.parent / "resources" / "dat"
            pkg_dat_path = pkg_dir / f"vocab_{name}.dat"
            pkg_json_path = pkg_dir / f"vocab_{name}.json"
            
        vocab = cls()
        
        # 1. Try Package Directory (Fastest & Most Reliable)
        if pkg_dat_path.exists() and _C_BACKEND_AVAILABLE:
            vocab._load_binary_dat(pkg_dat_path)
            if pkg_json_path.exists():
                 vocab._load_json_mappings(pkg_json_path) # For decoding
            return vocab
            
        # PATH 2: User Cache Directory (Development / Updates)
        cache_dir = Path.home() / ".cache" / "xerv" / "crayon" / "profiles"
        cache_dat_path = cache_dir / f"vocab_{name}.dat"
        cache_json_path = cache_dir / f"vocab_{name}.json"

        # 2. Try Cache Directory
        if cache_dat_path.exists() and _C_BACKEND_AVAILABLE:
            vocab._load_binary_dat(cache_dat_path)
            if cache_json_path.exists():
                 vocab._load_json_mappings(cache_json_path)
            return vocab

        # 3. JSON Fallback (Slow)
        if pkg_json_path.exists():
            print(f"[Crayon] DAT missing or Engine unavailable. Loading JSON {name} from package...")
            vocab._load_json_legacy(pkg_json_path)
            return vocab
        elif cache_json_path.exists():
            print(f"[Crayon] DAT missing or Engine unavailable. Loading JSON {name} from cache...")
            vocab._load_json_legacy(cache_json_path)
            return vocab

        # 4. Total Fallback: Build on Demand
        print(f"[Crayon] Profile {name} not found in package or cache. Building...")
        from ..resources import build_and_cache_profile
        build_and_cache_profile(name)
        
        # Reload from cache after build
        if cache_dat_path.exists() and _C_BACKEND_AVAILABLE:
             vocab._load_binary_dat(cache_dat_path)
        else:
             vocab._load_json_legacy(cache_json_path)
            
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
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not self.id_to_token:
            raise RuntimeError("Cannot decode: vocabulary mappings not loaded. "
                             "This may happen if using pure DAT mode without JSON mappings.")
        
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_token:
                tokens.append(self.id_to_token[token_id])
            else:
                tokens.append("<UNK>")  # Unknown token fallback
        
        return "".join(tokens)

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