"""
Crayon Resources Module.

Manages external data sources for 'batteries-included' vocabulary generation.
Strictly enforces streaming-only access to prevent local disk usage.

Data Sources (Priority: Local Files -> External Streaming):
1. Xerv-AI/RainDrop-DTS (General Instruction Following)
2. Xerv-AI/Physics-dataset-700 (Scientific Reasoning)
3. Xerv-AI/GRAD (Graduate Level Mathematics)
4. Tiny Shakespeare (Classical Literature/English)
"""

import logging
import csv
import json
from pathlib import Path
from typing import Iterator, List, Optional
from itertools import chain

# Configure module logger
logger = logging.getLogger(__name__)

# Optional imports - don't crash if not installed
try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False
    logger.debug("requests not installed - HTTP streaming disabled")

try:
    from datasets import load_dataset
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    logger.debug("datasets not installed - HuggingFace streaming disabled")


# ============================================================================
# Configuration for Data Sources
# ============================================================================

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

# Hugging Face Datasets Config: (Dataset Name, Split, List of Text Columns)
HF_SOURCES: List[tuple] = [
    ("Xerv-AI/RainDrop-DTS", "train", ["text"]),
    ("Xerv-AI/Physics-dataset-700", "train", ["Question", "Answer", "Reasoning"]),
    ("Xerv-AI/GRAD", "train", ["question", "solution"]),
]

# Chunk size for streaming (64KB fits L2 cache)
STREAM_CHUNK_SIZE = 65536

# Local resource directory
RESOURCE_DIR = Path(__file__).parent / "resources"


# ============================================================================
# Local Resource Iterators (Priority 1)
# ============================================================================

def yield_local_resources(max_grad_entries: int = 5000) -> Iterator[str]:
    """
    Yields text from local resource files if they exist.
    
    Files:
    - input.txt (Shakespeare)
    - data.csv (RainDrop-DTS)
    - physics_detailed_dataset_700_rows.csv (Physics)
    - graduate_math.jsonl (GRAD)
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

    # 2. RainDrop-DTS (CSV)
    raindrop_path = RESOURCE_DIR / "data.csv"
    if raindrop_path.exists():
        logger.info(f"Using local RainDrop-DTS: {raindrop_path}")
        try:
            with open(raindrop_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if 'text' in row and row['text']:
                        yield row['text']
        except Exception as e:
            logger.warning(f"Error reading local RainDrop-DTS: {e}")

    # 3. Physics (CSV)
    physics_path = RESOURCE_DIR / "physics_detailed_dataset_700_rows.csv"
    if physics_path.exists():
        logger.info(f"Using local Physics dataset: {physics_path}")
        try:
            with open(physics_path, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for col in ['Question', 'Answer', 'Reasoning']:
                        if col in row and row[col]:
                            yield row[col]
        except Exception as e:
            logger.warning(f"Error reading local Physics dataset: {e}")

    # 4. GRAD (JSONL)
    grad_path = RESOURCE_DIR / "graduate_math.jsonl"
    if grad_path.exists():
        logger.info(f"Using local GRAD dataset: {grad_path} (limit {max_grad_entries})")
        count = 0
        try:
            with open(grad_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if count >= max_grad_entries:
                        break
                    if line.strip():
                        try:
                            data = json.loads(line)
                            if 'question' in data:
                                yield data['question']
                                count += 1
                            if 'solution' in data and count < max_grad_entries:
                                # Truncate extremely long solutions to avoid OOM
                                solution = data['solution']
                                yield solution[:2000] if len(solution) > 2000 else solution
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.warning(f"Error reading local GRAD dataset: {e}")


# ============================================================================
# Streaming Iterators (Priority 2)
# ============================================================================

def yield_shakespeare_stream() -> Iterator[str]:
    """
    Streams Karpathy's Tiny Shakespeare directly from GitHub.
    Yields text in 64KB chunks.
    """
    if (RESOURCE_DIR / "input.txt").exists():
        return # Prefer local
        
    if not _REQUESTS_AVAILABLE:
        logger.warning("requests not installed - skipping Shakespeare source")
        return
    
    try:
        logger.info(f"Streaming source: {SHAKESPEARE_URL}")
        
        with requests.get(SHAKESPEARE_URL, stream=True, timeout=30) as response:
            response.raise_for_status()
            encoding = response.encoding or 'utf-8'
            buffer = ""
            for chunk in response.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                if chunk:
                    try:
                        text = chunk.decode(encoding)
                    except UnicodeDecodeError:
                        text = chunk.decode('utf-8', errors='ignore')
                    buffer += text
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            yield line
            if buffer.strip():
                yield buffer
                
        logger.info("Shakespeare stream completed")
        
    except Exception as e:
        logger.warning(f"Failed to stream Shakespeare: {e}")


def yield_hf_stream(
    dataset_name: str, 
    split: str, 
    columns: List[str],
    max_samples: Optional[int] = None
) -> Iterator[str]:
    """
    Streams a Hugging Face dataset without downloading to disk.
    """
    # Check if we have local equivalents roughly based on name mapping
    is_local_available = False
    if "RainDrop" in dataset_name and (RESOURCE_DIR / "data.csv").exists():
        is_local_available = True
    elif "Physics" in dataset_name and (RESOURCE_DIR / "physics_detailed_dataset_700_rows.csv").exists():
        is_local_available = True
    elif "GRAD" in dataset_name and (RESOURCE_DIR / "graduate_math.jsonl").exists():
        is_local_available = True
        
    if is_local_available:
        return # Skip streaming if local is present

    if not _HF_AVAILABLE:
        logger.warning(f"datasets not installed - skipping {dataset_name}")
        return

    try:
        logger.info(f"Streaming HF Source: {dataset_name} (columns: {columns})")
        ds = load_dataset(dataset_name, split=split, streaming=True, trust_remote_code=True)
        
        sample_count = 0
        for row in ds:
            for col in columns:
                content = row.get(col)
                if content:
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        for item in content:
                            if isinstance(item, str):
                                yield item
                            elif isinstance(item, dict):
                                for v in item.values():
                                    if isinstance(v, str):
                                        yield v
            sample_count += 1
            if max_samples and sample_count >= max_samples:
                break
        
        logger.info(f"Completed streaming {dataset_name}: {sample_count} samples")
                        
    except Exception as e:
        logger.warning(f"Failed to stream {dataset_name}: {e}")


def yield_builtin_corpus() -> Iterator[str]:
    """Yields a built-in corpus for minimal vocabulary construction."""
    # Common English words and patterns for baseline coverage
    builtin_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Pack my box with five dozen liquor jugs.",
        "How vexingly quick daft zebras jump!",
        "The five boxing wizards jump quickly.",
        "Sphinx of black quartz, judge my vow.",
        "Two driven jocks help fax my big quiz.",
        "The jay, pig, fox, zebra and my wolves quack!",
        "Sympathizing would fix Quaker objectives.",
        "A wizard's job is to vex chumps quickly in fog.",
        "Watch Jeopardy, Alex Trebek's fun TV quiz game.",
    ]
    
    # Programming/technical terms
    technical_texts = [
        "def function(self, param): return value",
        "class TokenizerClass: pass",
        "import numpy as np",
        "for i in range(100): print(i)",
        "if condition: do_something()",
        "try: result = compute() except: handle_error()",
        "async def fetch_data(): await response",
        "lambda x: x * 2",
        "with open('file.txt') as f: data = f.read()",
        "@decorator def decorated(): pass",
    ]
    
    # Numbers and special patterns
    pattern_texts = [
        "0123456789",
        "Hello, World! How are you?",
        "user@email.com",
        "https://example.com/path?query=value",
        "2024-01-15T12:30:45Z",
        "$1,234.56 USD",
        "Temperature: 98.6°F (37°C)",
        "Version 2.0.1-beta",
        "ISBN: 978-0-123456-78-9",
        "Phone: +1 (555) 123-4567",
    ]
    
    for text in builtin_texts + technical_texts + pattern_texts:
        yield text
    
    # Generate additional patterns
    for i in range(1000):
        yield f"token{i} word{i} item{i}"


def get_default_corpus_iterator(
    include_shakespeare: bool = True,
    include_hf_sources: bool = True,
    include_builtin: bool = True,
    max_hf_samples: Optional[int] = None
) -> Iterator[str]:
    """
    Returns a unified iterator over ALL default datasets.
    Prioritizes local files if present, otherwise streams.
    """
    iterators: List[Iterator[str]] = []
    
    # 1. Start with local resources (fastest, most reliable)
    iterators.append(yield_local_resources())
    
    # 2. Add built-in corpus (baseline fallback)
    if include_builtin:
        iterators.append(yield_builtin_corpus())
    
    # 3. Add Shakespeare (Stream if not local)
    if include_shakespeare:
        iterators.append(yield_shakespeare_stream())
    
    # 4. Add Hugging Face Sources (Stream if not local)
    if include_hf_sources:
        for name, split, cols in HF_SOURCES:
            iterators.append(yield_hf_stream(name, split, cols, max_hf_samples))

    # Chain all iterators sequentially
    return chain(*iterators)


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
