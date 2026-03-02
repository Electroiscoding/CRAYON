# Crayon v5.1.0 Demo

from crayon import CrayonVocab

tokenizer = CrayonVocab(device="auto")

print("--- Testing Standard ---")
tokenizer.load_profile("standard")
tokens_std = tokenizer.tokenize("that is a test for the standard profile and lite profile and god")
print(tokens_std)
print(tokenizer.decode(tokens_std))

print("--- Testing Lite ---")
tokenizer.load_profile("lite")
tokens_lite = tokenizer.tokenize("my daughter")
print(tokens_lite)
print(tokenizer.decode(tokens_lite))