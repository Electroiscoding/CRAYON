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
        min_frequency=5,  # Aggressive pruning for speed
        description="Ultra-lightweight for mobile/edge (English & Basic Logic)",
        sources=[
            ("wikitext", "train", ["text"]),
            ("Xerv-AI/RainDrop-DTS", "train", ["text"])
        ]
    ),
    "science": VocabProfile(
        name="science",
        target_size=250000,
        min_frequency=3,
        description="High-Precision Math, Physics & LaTeX Support",
        sources=[
            ("Xerv-AI/GRAD", "train", ["question", "solution"]),
            ("Xerv-AI/Physics-dataset-700", "train", ["Question", "Answer", "Reasoning"]),
            ("math_dataset", "train", ["question", "answer"]) 
        ]
    ),
    "code": VocabProfile(
        name="code",
        target_size=250000,
        min_frequency=2,
        description="Software Engineering (Python, Rust, C++, JS)",
        sources=[
            ("codeparrot/codeparrot-clean", "train", ["content"]),
            ("bigcode/the-stack-smol", "train", ["content"])
        ]
    ),
    "multilingual": VocabProfile(
        name="multilingual",
        target_size=250000,
        min_frequency=2,
        description="Global Language Support (European + Asian + Indic)",
        sources=[
            ("oscar-corpus/OSCAR-2201", "train", ["text"]), # Subset
            ("wikipedia", "train", ["text"])
        ]
    ),
    "arts_commerce": VocabProfile(
        name="arts_commerce",
        target_size=250000,
        min_frequency=2,
        description="Literature, Financial Reports, Legal & Business",
        sources=[
            ("pg19", "train", ["text"]), # Project Gutenberg
            ("financial_phrasebank", "train", ["sentence"]),
            ("multi_eurlex", "train", ["text"])
        ]
    )
}
