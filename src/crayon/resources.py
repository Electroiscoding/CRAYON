"""
Crayon Resources Module.

Manages external data sources for 'batteries-included' vocabulary generation.
Strictly enforces streaming-only access to prevent local disk usage.

Data Sources:
1. Xerv-AI/RainDrop-DTS (General Instruction Following)
2. Xerv-AI/Physics-dataset-700 (Scientific Reasoning)
3. Xerv-AI/GRAD (Graduate Level Mathematics)
4. Tiny Shakespeare (Classical Literature/English)
"""

import logging
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


# ============================================================================
# Streaming Iterators
# ============================================================================

def yield_shakespeare_stream() -> Iterator[str]:
    """
    Streams Karpathy's Tiny Shakespeare directly from GitHub.
    
    Uses chunked streaming to minimize memory footprint.
    Yields text in 64KB chunks.
    """
    if not _REQUESTS_AVAILABLE:
        logger.warning("requests not installed - skipping Shakespeare source")
        return
    
    try:
        logger.info(f"Streaming source: {SHAKESPEARE_URL}")
        
        with requests.get(SHAKESPEARE_URL, stream=True, timeout=30) as response:
            response.raise_for_status()
            
            # Detect encoding from response or default to UTF-8
            encoding = response.encoding or 'utf-8'
            
            # Stream in chunks to minimize memory
            buffer = ""
            for chunk in response.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                if chunk:
                    # Decode chunk
                    try:
                        text = chunk.decode(encoding)
                    except UnicodeDecodeError:
                        text = chunk.decode('utf-8', errors='ignore')
                    
                    buffer += text
                    
                    # Yield complete lines to avoid mid-word splits
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        if line.strip():
                            yield line
            
            # Yield remaining buffer
            if buffer.strip():
                yield buffer
                
        logger.info("Shakespeare stream completed")
        
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to stream Shakespeare: {e}")
    except Exception as e:
        logger.error(f"Unexpected error streaming Shakespeare: {e}")


def yield_hf_stream(
    dataset_name: str, 
    split: str, 
    columns: List[str],
    max_samples: Optional[int] = None
) -> Iterator[str]:
    """
    Streams a Hugging Face dataset without downloading to disk.
    
    Uses 'streaming=True' to fetch data on-the-fly with minimal memory.
    
    Args:
        dataset_name: HuggingFace dataset identifier
        split: Dataset split (train/test/validation)
        columns: List of column names to extract text from
        max_samples: Optional limit on number of samples
        
    Yields:
        Text strings from the specified columns
    """
    if not _HF_AVAILABLE:
        logger.warning(f"datasets not installed - skipping {dataset_name}")
        return

    try:
        logger.info(f"Streaming HF Source: {dataset_name} (columns: {columns})")
        
        # streaming=True prevents local download/caching
        ds = load_dataset(
            dataset_name, 
            split=split, 
            streaming=True,
            trust_remote_code=True
        )
        
        sample_count = 0
        for row in ds:
            # Extract text from specified columns
            for col in columns:
                content = row.get(col)
                if content:
                    if isinstance(content, str):
                        yield content
                    elif isinstance(content, list):
                        # Handle conversation/list formats
                        for item in content:
                            if isinstance(item, str):
                                yield item
                            elif isinstance(item, dict):
                                # Handle dict items (e.g., {"role": "user", "content": "..."})
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
    """
    Yields a built-in corpus for minimal vocabulary construction.
    
    Used as fallback when external sources are unavailable.
    Provides basic English text coverage.
    """
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
    
    # Common word combinations
    common_words = [
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her", "she",
        "or", "an", "will", "my", "one", "all", "would", "there", "their", "what",
    ]
    
    for i, word1 in enumerate(common_words):
        for word2 in common_words[i:i+5]:
            yield f"{word1} {word2}"


def get_default_corpus_iterator(
    include_shakespeare: bool = True,
    include_hf_sources: bool = True,
    include_builtin: bool = True,
    max_hf_samples: Optional[int] = None
) -> Iterator[str]:
    """
    Returns a unified iterator over ALL default datasets.
    
    Composition (in order):
    1. Built-in corpus (always available, baseline coverage)
    2. Tiny Shakespeare (Literature/English)
    3. Xerv-AI/RainDrop-DTS (General Instruction)
    4. Xerv-AI/Physics-dataset-700 (Scientific Reasoning)
    5. Xerv-AI/GRAD (Graduate Level Mathematics)
    
    Args:
        include_shakespeare: Include Tiny Shakespeare source
        include_hf_sources: Include HuggingFace datasets
        include_builtin: Include built-in fallback corpus
        max_hf_samples: Limit samples per HF dataset (for testing)
        
    Returns:
        Iterator yielding text strings from all sources
    """
    iterators: List[Iterator[str]] = []
    
    # 1. Add built-in corpus (always available as baseline)
    if include_builtin:
        iterators.append(yield_builtin_corpus())
    
    # 2. Add Shakespeare
    if include_shakespeare and _REQUESTS_AVAILABLE:
        iterators.append(yield_shakespeare_stream())
    
    # 3. Add Hugging Face Sources
    if include_hf_sources and _HF_AVAILABLE:
        for name, split, cols in HF_SOURCES:
            iterators.append(yield_hf_stream(name, split, cols, max_hf_samples))
    elif include_hf_sources:
        logger.warning(
            "HuggingFace datasets not available. "
            "Install with: pip install xerv-crayon[full]"
        )

    # Chain all iterators sequentially
    return chain(*iterators)


def check_resource_availability() -> dict:
    """
    Check which data sources are available.
    
    Returns:
        Dict with availability status for each source type
    """
    return {
        "requests_available": _REQUESTS_AVAILABLE,
        "huggingface_available": _HF_AVAILABLE,
        "shakespeare_url": SHAKESPEARE_URL,
        "hf_sources": [name for name, _, _ in HF_SOURCES],
        "builtin_available": True
    }
