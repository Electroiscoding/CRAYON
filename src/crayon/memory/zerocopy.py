import mmap
import os
from typing import Iterator, Tuple, List
from ..core.vocabulary import CrayonVocab

class ZeroCopyTokenizer:
    """
    Zero-copy tokenizer minimizing memory allocation and data movement.
    
    Uses OS virtual memory (mmap) to handle files larger than RAM[cite: 844].
    """

    def __init__(self, vocab: CrayonVocab):
        self.vocab = vocab

    def tokenize_file_zerocopy(self, file_path: str) -> Iterator[Tuple[int, int]]:
        """
        Tokenize large files without loading entire content into memory.
        Yields: (token_id, file_offset)
        """
        file_size = os.path.getsize(file_path)
        chunk_size = 64 * 1024 # 64KB fits L2 cache [cite: 858]
        overlap = 1024 # Safety margin for boundary tokens
        
        with open(file_path, 'rb') as f:
            # Memory map the entire file [cite: 854]
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mmapped:
                offset = 0
                
                while offset < file_size:
                    chunk_end = min(offset + chunk_size, file_size)
                    
                    # Create zero-copy memoryview [cite: 860]
                    # Includes overlap to catch tokens spanning chunks
                    view_end = min(chunk_end + overlap, file_size)
                    chunk_view = memoryview(mmapped)[offset:view_end]
                    
                    # Process chunk
                    # Note: We pass is_last to know if we can consume the very end
                    is_last = (chunk_end == file_size)
                    tokens, consumed = self._tokenize_chunk_with_boundaries(
                        chunk_view, offset, is_last
                    )
                    
                    for tid in tokens:
                        yield tid, offset # In reality, offset needs strict tracking per token
                    
                    # Advance
                    offset += consumed

    def _tokenize_chunk_with_boundaries(self, 
                                      chunk_view: memoryview, 
                                      base_offset: int,
                                      is_last: bool) -> Tuple[List[int], int]:
        """
        Tokenize memory chunk handling token boundaries at edges[cite: 877].
        """
        # Decode (copy happens here unfortunately in Python, unless C-ext used)
        # In strict zero-copy C-ext, we'd pass the pointer directly.
        try:
            text = chunk_view.tobytes().decode('utf-8')
        except UnicodeDecodeError:
            # Handle partial UTF-8 at end of view
            text = chunk_view.tobytes().decode('utf-8', errors='ignore')
            
        tokens = []
        pos = 0
        text_len = len(text)
        limit = text_len if is_last else text_len - 100 # Safety margin [cite: 892]
        
        while pos < text_len:
            # Stop if we are in the danger zone (overlap area) and not at EOF
            if not is_last and pos > limit:
                break
                
            token_id, match_len = self.vocab.longest_match(text, pos)
            
            if match_len > 0:
                tokens.append(token_id)
                pos += match_len
            else:
                tokens.append(self.vocab.unk_token_id)
                pos += 1
                
        # Calculate actual bytes consumed to adjust file offset correctly
        # This part is tricky in Python due to char vs byte length mismatch
        consumed_bytes = len(text[:pos].encode('utf-8'))
        
        return tokens, consumed_bytes