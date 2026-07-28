from crayon import CrayonVocab

tokenizer = CrayonVocab(device="auto")
tokenizer.load_profile("standard")

print(f"--- Turbo Engine: {tokenizer._turbo_backend.get_hardware_info() if tokenizer._turbo_backend else tokenizer.device} ---")

print("\n--- Testing Standard Profile ---")
tokens_std = tokenizer.tokenize("that is a test for the standard profile and lite profile and god")
print(f"Tokens: {tokens_std}")
print(f"Decoded: {tokenizer.decode(tokens_std)}")

print("\n--- Testing Lite Profile ---")
tokenizer.load_profile("lite")
tokens_lite = tokenizer.tokenize("my daughter")
print(f"Tokens: {tokens_lite}")
print(f"Decoded: {tokenizer.decode(tokens_lite)}")

print("\n--- Direct Turbo C Engine (Zero-Copy NumPy Output) ---")
if tokenizer._turbo_backend:
    np_tokens = tokenizer._turbo_backend.tokenize("Ultra-fast tokenization with Crayon Turbo")
    print(f"Direct Turbo Tokens ({type(np_tokens).__name__}): {np_tokens}")