"""
Train Vocabulary from Xerv-AI Datasets.

Streams data from:
- Xerv-AI/GRAD (Graduate Mathematics)
- Xerv-AI/Physics-dataset-700 (Scientific Reasoning)
- Xerv-AI/RainDrop-DTS (General Instruction)
- Tiny Shakespeare (Classical Literature)
"""

import logging
import time
from crayon import CrayonVocab

# Enable logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def progress_callback(msg: str):
    print(f"[PROGRESS] {msg}")

def main():
    print("=" * 60)
    print("XERV Crayon Vocabulary Training")
    print("=" * 60)
    
    print("\nStreaming from Xerv-AI datasets...")
    print("  - Xerv-AI/RainDrop-DTS")
    print("  - Xerv-AI/Physics-dataset-700")
    print("  - Xerv-AI/GRAD")
    print("  - Tiny Shakespeare")
    print()
    
    start_time = time.time()
    
    # Build vocabulary from default sources (your datasets)
    vocab = CrayonVocab.from_default_sources(
        vocab_size=50000,  # Start with 50k for faster training
        progress_callback=progress_callback
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n[DONE] Vocabulary built in {elapsed:.1f}s")
    print(f"       Vocabulary size: {len(vocab)} tokens")
    print(f"       C-Extension: {'Enabled' if vocab._c_ext_available else 'Disabled'}")
    
    # Save vocabulary
    vocab.save("trained_vocab.json", format="json")
    vocab.save("trained_vocab.txt", format="txt")
    print(f"\n[SAVED] trained_vocab.json")
    print(f"[SAVED] trained_vocab.txt")
    
    # Test it
    print("\n" + "=" * 60)
    print("Testing Trained Vocabulary")
    print("=" * 60)
    
    test_texts = [
        "delhi is the capital of india",
        "The quick brown fox jumps over the lazy dog",
        "What is the derivative of x squared?",
        "Calculate the force using F = ma",
    ]
    
    for text in test_texts:
        tokens = vocab.tokenize(text)
        decoded = vocab.decode(tokens)
        print(f"\nInput:   '{text}'")
        print(f"Tokens:  {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"Decoded: '{decoded[:50]}{'...' if len(decoded) > 50 else ''}'")

if __name__ == "__main__":
    main()
