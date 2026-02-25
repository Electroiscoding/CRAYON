# Crayon v5.0.1 - Omni-Backend Tokenizer Final Demo
# Installation from TestPyPI:
# !pip install -i https://test.pypi.org/simple/ xerv-crayon==5.0.1

from crayon import CrayonVocab

# Device auto-detection (CPU/CUDA/ROCm)
tokenizer = CrayonVocab(device="auto")

print("--- Testing Code ---")
tokenizer.load_profile("code")
tokens_code = tokenizer.tokenize("t fetc")
print(f"Tokens: {tokens_code}")
print(f"Decoded: {tokenizer.decode(tokens_code)}")

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

print("\n--- Testing Multilingual ---")
tokenizer.load_profile("multilingual")
tokens_multi = tokenizer.tokenize("द")
print(f"Tokens: {tokens_multi}")
print(f"Decoded: {tokenizer.decode(tokens_multi)}")
