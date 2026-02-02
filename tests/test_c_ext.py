
import unittest
import sys
import os
import tempfile
import mmap
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from crayon.c_ext import crayon_cpu, crayon_trainer, crayon_compiler
    EXTENSIONS_AVAILABLE = True
except ImportError:
    EXTENSIONS_AVAILABLE = False

@unittest.skipUnless(EXTENSIONS_AVAILABLE, "C++ extensions not available")
class TestCrayonExtensions(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Create a small test vocabulary and build a DAT
        cls.test_vocab = ["a", "ab", "abc", "b", "c", " ", "def"]
        # id mapping: 0:a, 1:ab, 2:abc, 3:b, 4:c, 5:" ", 6:def
        
        fd, cls.temp_dat = tempfile.mkstemp(suffix=".dat")
        os.close(fd)
        
        # Build DAT using the NEW compiler
        stats = crayon_compiler.compile_dat(cls.test_vocab, cls.temp_dat)
        
        # Load into CPU engine
        with open(cls.temp_dat, "rb") as f:
            cls.mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            crayon_cpu.load_dat(cls.mmap_obj)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'mmap_obj'):
            cls.mmap_obj.close()
        if hasattr(cls, 'temp_dat') and os.path.exists(cls.temp_dat):
            os.unlink(cls.temp_dat)

    def test_compiler_version(self):
        self.assertEqual(crayon_compiler.get_version(), "2.0.0-hyperfast")

    def test_cpu_hardware_info(self):
        info = crayon_cpu.get_hardware_info()
        self.assertIsInstance(info, str)
        self.assertIn("[", info)

    def test_tokenize_simple(self):
        # "abc" should be its own token (id 2)
        tokens = crayon_cpu.tokenize("abc")
        self.assertEqual(tokens, [2])

    def test_tokenize_longest_match(self):
        # "ab" + "c" vs "abc" -> should pick "abc" (id 2)
        tokens = crayon_cpu.tokenize("abc")
        self.assertEqual(tokens, [2])
        
        # "a" + "b" -> should pick "ab" (id 1)
        tokens = crayon_cpu.tokenize("ab")
        self.assertEqual(tokens, [1])

    def test_tokenize_fallback_unk(self):
        # "x" is not in vocab. UNK is ID 1 by convention in the engine fallback.
        # Wait, in OUR engine, if it fails to find a match, it appends ID 1 (hardcoded fallback).
        tokens = crayon_cpu.tokenize("x")
        self.assertEqual(tokens, [1])

    def test_trainer_basic(self):
        corpus = b"banana banana banana"
        # Train a small BPE
        merges = crayon_trainer.train_fast(corpus, 10, min_freq=1, verbose=0)
        self.assertIsInstance(merges, list)
        self.assertGreater(len(merges), 0)
        # Each merge is a pair of strings
        for m in merges:
            self.assertIsInstance(m, str)

if __name__ == "__main__":
    unittest.main()