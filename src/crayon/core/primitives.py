import dataclasses

@dataclasses.dataclass(slots=True, frozen=True)
class TokenMetadata:
    """
    Slots-based dataclass eliminates dictionary overhead.
    Frozen=True enables additional optimizations in Python 3.12+.
    
    Memory Layout:
    - token_id (int): 28 bytes
    - frequency (int): 28 bytes
    - average_length (float): 24 bytes
    Total per instance overhead is minimal compared to standard class.
    """
    token_id: int
    frequency: int
    average_length: float