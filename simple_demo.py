# Crayon v5.1.0 Simple Demo

from crayon import CrayonVocab

def main():
    print("Crayon Tokenizer Demo")
    print("=======================\n")

    # 1. Initialize & Load Profile
    # 'auto' will use GPU if available, else CPU
    vocab = CrayonVocab(device="auto")
    vocab.load_profile("lite") 
    print(f"Loaded Profile: 'lite' on {vocab.device.upper()}")

    # 2. Define Input Text
    text = "Hello, Crayon! This is a simple test."

    # 3. Tokenize
    # This converts the string into a list of integer IDs
    tokens = vocab.tokenize(text)

    print(f"\nInput Text:  '{text}'")
    print(f"Token IDs:   {tokens}")
    print(f"Count:       {len(tokens)} tokens\n")

    # 4. Analyze Each Token
    # We decode each ID individually to show exactly what substring it represents
    print("Token Breakdown:")
    print(f"{'ID':<8} | {'Substring':<20}")
    print("-" * 30)

    for tid in tokens:
        # We pass a list [tid] because decode expects a sequence
        substring = vocab.decode([tid])
        print(f"{tid:<8} | '{substring}'")

    # 5. Full Decode
    # Convert the list of IDs back to the original string
    decoded_text = vocab.decode(tokens)
    print(f"\nFull Decode check: '{decoded_text}'")
    
    # Verification
    if text == decoded_text:
        print("[MATCH] Exact Match!")
    else:
        print("[MISMATCH] Mismatch (canonicalization might differ)")

if __name__ == "__main__":
    main()
