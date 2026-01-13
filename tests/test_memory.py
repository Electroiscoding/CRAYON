import unittest
import os
import gc
import tempfile
from crayon.memory.pool import MemoryPool
from crayon.memory.zerocopy import ZeroCopyTokenizer
from crayon.core.vocabulary import CrayonVocab

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
        
        # Get it back (should be same object or at least count is correct)
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
                
            self.assertEqual(count, 2000)  # 1000 "test" + 1000 " "
        finally:
            # Ensure all references are released before deleting (Windows mmap issue)
            gc.collect()
            try:
                os.remove(fname)
            except PermissionError:
                pass  # Windows may still hold file, ignore cleanup failure

    def test_pool_oversized_buffer(self):
        """Test that oversized buffers are not pooled."""
        pool = MemoryPool(chunk_size=1024, pool_size=2)
        
        # Request larger buffer
        big_buf = pool.get_buffer(required_size=4096)
        self.assertEqual(len(big_buf), 4096)
        
        # Return it - should not be added to pool
        pool.return_buffer(big_buf)
        self.assertEqual(len(pool.available_buffers), 2)  # Original pool unchanged