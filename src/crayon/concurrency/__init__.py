"""
Crayon Concurrency Module.

This module implements the high-throughput parallelization strategies described in
Section 7 of the XERV Crayon Engineering Treatise. It includes:
1. Pipeline Architecture (Instruction-level parallelism concept applied to tokenization)
2. Thread-Local Isolation (GIL-aware resource management)
"""

from .pipeline import PipelineTokenizer
from .thread_local import ThreadLocalTokenizer

__all__ = ["PipelineTokenizer", "ThreadLocalTokenizer"]