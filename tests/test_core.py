import unittest
from src.crayon.core.vocabulary import CrayonVocab
from src.crayon.core.primitives import TokenMetadata

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
        text = "unfortunatxely" # 'x' is unknown
        ids = self.vocab.tokenize(text)
        
        # unfortunat (partial match? No, strict prefix)
        # 'unfortunate' mismatch at 'x'. 
        # Logic: 
        # 'unfortunate' matches 'unfortunat'? No.
        # 'un' matches.
        # Remainder: 'fortunatxely'
        # ... fallback sequence
        
        # Simplified check for presence of UNK
        self.assertIn(self.vocab.unk_token_id, ids)

    def test_metadata_memory_layout(self):
        """Verify primitives use slots."""
        meta = TokenMetadata(token_id=1, frequency=100, average_length=5.5)
        with self.assertRaises(AttributeError):
            meta.new_attr = 1  # Should fail due to __slots__