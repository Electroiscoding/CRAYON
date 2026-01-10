# `TokenMetadata` frozen slots dataclass
from dataclasses import dataclass

@dataclass(frozen=True)
class TokenMetadata:
    __slots__ = ['id', 'text', 'score']
    id: int
    text: str
    score: float
