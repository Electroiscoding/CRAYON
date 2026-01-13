import unittest
from crayon.core.vocabulary import CrayonVocab
from crayon.core.primitives import TokenMetadata

class TestCoreTokenization(unittest.TestCase):
    
    def setUp(self):
        self.tokens = ["un", "fortunate", "ly", "unfortunate", "man"]
        self.vocab = CrayonVocab(self.tokens, unk_token="<UNK>")

    def test_longest_match_priority(self):
        """
        Verify that the tokenizer strictly prefers the longest match.
        'unfortunately' -> 'unfortunate' + 'ly' (if 'unfortunately' not in vocab)
        """
        text = "unfortunately"
        ids = self.vocab.tokenize(text)
        resolved_tokens = [self.vocab.id_to_token[i] for i in ids]
        
        # 'unfortunate' is in vocab, so it should be picked over 'un' + 'fortunate'
        self.assertEqual(resolved_tokens, ["unfortunate", "ly"])

    def test_unknown_token_fallback(self):
        """Verify <UNK> handling."""
        text = "unfortunatxely"  # 'x' is unknown
        ids = self.vocab.tokenize(text)
        
        # Simplified check for presence of UNK
        self.assertIn(self.vocab.unk_token_id, ids)

    def test_metadata_memory_layout(self):
        """Verify primitives use slots."""
        meta = TokenMetadata(token_id=1, frequency=100, average_length=5.5)
        # Frozen dataclasses raise FrozenInstanceError (Python 3.10+) or TypeError
        with self.assertRaises((AttributeError, TypeError)):
            meta.new_attr = 1  # Should fail due to __slots__ and frozen=True

    def test_vocabulary_contains(self):
        """Test vocabulary membership checks."""
        self.assertIn("unfortunate", self.vocab)
        self.assertNotIn("nonexistent", self.vocab)

    def test_vocabulary_size(self):
        """Test vocabulary size."""
        self.assertEqual(len(self.vocab), 5)

    def test_decode(self):
        """Test decoding token IDs back to string."""
        ids = [3, 2]  # "unfortunate" + "ly"
        decoded = self.vocab.decode(ids)
        self.assertEqual(decoded, "unfortunately")