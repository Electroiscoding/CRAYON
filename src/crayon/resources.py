"""
Crayon Resources Module.
Manages atomic building and streaming for Vocabulary Profiles.
"""
import os
import json
import shutil
import logging
import csv
from pathlib import Path
from typing import Iterator, List, Optional
from itertools import chain

from .core.profiles import VocabProfile, PROFILES

# Configure module logger
logger = logging.getLogger(__name__)

# Optional imports
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False


# ============================================================================
# Profile Streaming and Caching
# ============================================================================

# Cache Configuration
CACHE_DIR = Path.home() / ".cache" / "xerv" / "crayon" / "profiles"

def get_profile_path(profile: VocabProfile) -> Path:
    """Returns versioned path: ~/.cache/.../vocab_science_v1.json"""
    return CACHE_DIR / f"vocab_{profile.name}_{profile.version}.json"

def yield_profile_stream(profile: VocabProfile, prefer_local_only: bool = False) -> Iterator[str]:
    """
    Resilient Streamer: Iterates through sources. 
    1. Checks for local sample/bootstrap corpus first.
    2. Streams from Hugging Face if available (unless prefer_local_only=True).
    """
    # 1. Local Bootstrap Corpus (Seamless Offline Fallback)
    # Checks for resources/science_corpus.txt, resources/code_corpus.txt, etc.
    # The convention is resources/{profile_name}_corpus.txt
    local_corpus_path = RESOURCE_DIR / f"{profile.name}_corpus.txt"
    has_local = False
    
    if local_corpus_path.exists():
        logger.info(f"[Sources] Found local bootstrap corpus: {local_corpus_path}")
        has_local = True
        try:
            with open(local_corpus_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield line.strip()
        except Exception as e:
            logger.warning(f"Failed to read local corpus {local_corpus_path}: {e}")
            
    # Also support specific overrides
    if profile.name == "lite":
        # Lite profile always includes Shakespeare & RainDrop from local if present
        yield from yield_local_resources()
        has_local = True

    # If we want to force local usage and we found local data, skip remote
    if prefer_local_only and has_local:
        logger.info(f"[Mode] Skipping remote sources for {profile.name} (Local-Only Build)")
        return

    # 2. Hugging Face Sources
    if not _HF_AVAILABLE:
        logger.info("HuggingFace 'datasets' not installed. Skipping remote sources.")
        return

    for ds_name, split, cols in profile.sources:
        try:
            logger.info(f"[Stream] Connecting to {ds_name}...")
            
            # Special handling for wikitext which requires a config name
            load_args = [ds_name]
            if ds_name == "wikitext":
                load_args.append("wikitext-103-v1")
                
            # Try loading with trust_remote_code=True first
            try:
                ds = load_dataset(*load_args, split=split, streaming=True, trust_remote_code=True)
            except Exception:
                # Fallback without trust_remote_code (some datasets forbid it)
                ds = load_dataset(*load_args, split=split, streaming=True, trust_remote_code=False)
            
            # Safety Cap: Process max 100k rows per source to prevent infinite hangs
            sample_count = 0
            for row in ds:
                if sample_count >= 100000: 
                    break 
                
                for col in cols:
                    val = row.get(col)
                    if isinstance(val, str): 
                        yield val
                    elif isinstance(val, list): 
                        # Handle list of strings (e.g. sentences)
                        yield " ".join(str(x) for x in val)
                
                sample_count += 1
                    
        except Exception as e:
            logger.warning(f"[Stream Warning] Failed to stream {ds_name}: {e}. Skipping source.")

def build_and_cache_profile(profile_name: str, prefer_local_only: bool = False) -> Path:
    """
    The Production Builder.
    1. Validates profile.
    2. Streams data (Zero-Disk).
    3. Trains entropy model.
    4. ATOMIC WRITE (Write tmp -> Rename) to prevent corruption.
    """
    # Lazy import to prevent circular dependency
    from .training import train_vocabulary 
    
    profile = PROFILES.get(profile_name)
    if not profile:
        raise ValueError(f"Unknown profile: '{profile_name}'. Available: {list(PROFILES.keys())}")

    target_path = get_profile_path(profile)
    
    # Fast Path: Return if already exists
    if target_path.exists():
        return target_path

    logger.info(f"--- BUILDING PROFILE: {profile.name.upper()} ---")
    logger.info(f"Target Size: {profile.target_size} | Sources: {len(profile.sources)}")
    
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Train
    stream = yield_profile_stream(profile, prefer_local_only=prefer_local_only)
    
    # If HF is not available or stream yields nothing, we might crash training.
    # But train_vocabulary handles iterators.
    vocab_list = train_vocabulary(
        stream, 
        target_size=profile.target_size,
        min_frequency=profile.min_frequency
    )
    
    # 2. Atomic Write Pattern
    temp_path = target_path.with_suffix(".tmp")
    try:
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_list, f, indent=2)
        
        # Instant rename (Atomic)
        shutil.move(str(temp_path), str(target_path))
        logger.info(f"[Success] Saved profile to: {target_path}")
        
    except Exception as e:
        if temp_path.exists(): 
            os.remove(temp_path)
        raise RuntimeError(f"Failed to save profile: {e}")
        
    return target_path


# ============================================================================
# Local Resource Iterators (Legacy / Fallback support)
# ============================================================================

RESOURCE_DIR = Path(__file__).parent / "resources"

def yield_local_resources(max_grad_entries: int = 5000) -> Iterator[str]:
    """
    Yields text from local resource files if they exist.
    """
    if not RESOURCE_DIR.exists():
        return

    # 1. Shakespeare
    shakespeare_path = RESOURCE_DIR / "input.txt"
    if shakespeare_path.exists():
        logger.info(f"Using local Shakespeare: {shakespeare_path}")
        try:
            with open(shakespeare_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        yield line.strip()
        except Exception as e:
            logger.warning(f"Error reading local Shakespeare: {e}")

def get_default_corpus_iterator(
    include_shakespeare: bool = True,
    include_hf_sources: bool = True, # Ignored in legacy shim
    include_builtin: bool = True,
    max_hf_samples: Optional[int] = None
) -> Iterator[str]:
    """
    Legacy shim: Returns an iterator over 'lite' profile resources or local.
    """
    # Prefer local resources first
    local_iter = yield_local_resources()
    
    # If no local resources, try to stream 'lite' profile if HF available
    if _HF_AVAILABLE:
        lite_profile = PROFILES.get("lite")
        if lite_profile:
            return chain(local_iter, yield_profile_stream(lite_profile))
            
    return local_iter

def check_resource_availability() -> dict:
    """Check which data sources are available."""
    local_files = [f.name for f in RESOURCE_DIR.iterdir()] if RESOURCE_DIR.exists() else []
    
    return {
        "requests_available": _REQUESTS_AVAILABLE,
        "huggingface_available": _HF_AVAILABLE,
        "local_resources_dir": str(RESOURCE_DIR),
        "local_files": local_files,
        "builtin_available": True
    }
