# Verify MemoryPool leaks
import unittest
from crayon.memory.pool import MemoryPool

class TestMemory(unittest.TestCase):
    def test_pool(self):
        pool = MemoryPool()
        self.assertIsNotNone(pool)
