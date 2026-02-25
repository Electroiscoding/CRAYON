from crayon import CrayonVocab

tokenizer = CrayonVocab(device="auto")

print("--- Testing Code ---")
tokenizer.load_profile("code")
tokens_lite = tokenizer.tokenize("t fetc")
print(tokens_lite)
print(tokenizer.decode(tokens_lite))

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

print("--- Testing Multilingual ---")
tokenizer.load_profile("multilingual")
tokens_lite = tokenizer.tokenize("द")
print(tokens_lite)
print(tokenizer.decode(tokens_lite))