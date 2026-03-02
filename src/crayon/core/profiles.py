"""
Crayon Profile Definitions.
Defines the 'Cartridges' available for the tokenizer ecosystem.
"""
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

@dataclass(frozen=True)
class VocabProfile:
    name: str
    target_size: int
    description: str
    # List of (Dataset_Name, Split, [Column_Names])
    sources: List[Tuple[str, str, List[str]]]
    min_frequency: int = 2
    version: str = "v1"

# --- The Production Cartridge Menu ---
PROFILES = {
    "lite": VocabProfile(
        name="lite",
        target_size=50000,
        min_frequency=0,
        description="Lite profile (tiktoken 50k)",
        sources=[]
    ),
    "standard": VocabProfile(
        name="standard",
        target_size=250000,
        min_frequency=0,
        description="Standard profile (tiktoken 50k + tiktoken 200k = 250k)",
        sources=[]
    ),
}
