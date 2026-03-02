# Crayon v5.1.0 - Omni-Backend Tokenizer Final Demo
# Installation:
# !pip install xerv-crayon==5.1.0

from crayon import CrayonVocab

# Device auto-detection (CPU/CUDA/ROCm)
tokenizer = CrayonVocab(device="auto")

print("\n--- Testing Standard ---")
tokenizer.load_profile("standard")
tokens_std = tokenizer.tokenize("that is a test for the standard profile and lite profile and god")
print(f"Tokens: {tokens_std}")
print(f"Decoded: {tokenizer.decode(tokens_std)}")

print("\n--- Testing Lite ---")
tokenizer.load_profile("lite")
tokens_lite = tokenizer.tokenize("my daughter")
print(f"Tokens: {tokens_lite}")
print(f"Decoded: {tokenizer.decode(tokens_lite)}")
