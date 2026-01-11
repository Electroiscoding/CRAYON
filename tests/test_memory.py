import unittest
import os
import tempfile
from src.crayon.memory.pool import MemoryPool
from src.crayon.memory.zerocopy import ZeroCopyTokenizer
from src.crayon.core.vocabulary import CrayonVocab

class TestMemorySubsystem(unittest.TestCase):
    
    def test_pool_recycling(self):
        """Verify buffers are actually returned to the pool."""
        pool = MemoryPool(chunk_size=1024, pool_size=2)
        
        # Get 2 buffers
        b1 = pool.get_buffer()
        b2 = pool.get_buffer()
        self.assertEqual(len(pool.available_buffers), 0)
        
        # Return 1
        pool.return_buffer(b1)
        self.assertEqual(len(pool.available_buffers), 1)
        
        # Get it back (should be same object in pure python pool impl usually, 
        # or at least count is correct)
        b3 = pool.get_buffer()
        self.assertEqual(len(pool.available_buffers), 0)

    def test_zerocopy_file_processing(self):
        """Verify memory mapped tokenization."""
        # Create dummy file
        with tempfile.NamedTemporaryFile(delete=False, mode='w', encoding='utf-8') as f:
            f.write("test " * 1000)
            fname = f.name
            
        try:
            vocab = CrayonVocab(["test", " "])
            zc = ZeroCopyTokenizer(vocab)
            
            count = 0
            for _ in zc.tokenize_file_zerocopy(fname):
                count += 1
                
            self.assertEqual(count, 2000) # 1000 "test" + 1000 " "
        finally:
            os.remove(fname)