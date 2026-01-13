from crayon import CrayonVocab
import sys

try:
    # Load the trained vocabulary
    v = CrayonVocab.from_json('trained_vocab.json')
    
    # Text to verify
    text = "delhi is india's capital"
    
    # Tokenize
    tokens = v.tokenize(text)
    decoded = v.decode(tokens)
    
    print("\n" + "="*40)
    print("FINAL VERIFICATION")
    print("="*40)
    print(f"Input:   {text}")
    print(f"Tokens:  {tokens}")
    print(f"Decoded: {decoded}")
    print("="*40 + "\n")
    
except Exception as e:
    print(f"Error: {e}")
