import json
import os
import sys

def build_standard():
    # Load lite vocab
    with open('trained_vocab_lite.json', 'r', encoding='utf-8') as f:
        lite_data = json.load(f)
    
    vocab_list = []
    if isinstance(lite_data, list):
        vocab_list = lite_data
    elif isinstance(lite_data, dict):
        vocab_list = [k for k, v in sorted(lite_data.items(), key=lambda x: x[1])]
    
    existing = set(vocab_list)
    print(f"Loaded {len(vocab_list)} tokens from lite.")
    
    # Load top 10000 words
    with open('top_10000_words.txt', 'r', encoding='utf-8') as f:
        words = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(words)} words from top 10000 list.")
    
    added_count = 0
    # Add Space prefix convention for standard words? The tokenizer might expect normal text.
    # We will just add the words as they appear, plus versions with leading space, to match standard subword tokenizers roughly.
    # The user requested "direct surgery" by merging top 10000 words.
    for w in words:
        if w not in existing:
            vocab_list.append(w)
            existing.add(w)
            added_count += 1
        # Also add capitalized and space-prefixed? The user didn't ask for that, let's keep it simple.
        
    print(f"Added {added_count} new unique words.")
    print(f"Total standard vocab size: {len(vocab_list)}")
    
    # Create V2 format dictionary
    vocab_dict = {"vocab": {}}
    for idx, word in enumerate(vocab_list):
        vocab_dict["vocab"][word] = idx
        
    out_dir = os.path.join('src', 'crayon', 'resources', 'dat')
    os.makedirs(out_dir, exist_ok=True)
    
    json_path = os.path.join(out_dir, 'vocab_standard.json')
    dat_path = os.path.join(out_dir, 'vocab_standard.dat')
    
    # Write JSON with proper indentation for "each word in new lines"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    print(f"Saved JSON to {json_path}")
    
    # Compile DAT using the hyper-fast C++ compiler
    try:
        from crayon.c_ext import crayon_compiler
        print("Using crayon_compiler to build DAT...")
        stats = crayon_compiler.compile_dat(vocab_list, dat_path)
        print("Compile stats:", stats)
    except Exception as e:
        print("Failed to use C++ compiler, falling back to python builder:", e)
        # Fallback to python DATBuilder
        from crayon.c_ext.dat_builder import DATBuilder
        builder = DATBuilder()
        builder.build(vocab_list)
        builder.save(dat_path)
    
    print(f"Successfully created Standard profile at {dat_path}")

if __name__ == '__main__':
    # Make sure we import the local crayon
    sys.path.insert(0, os.path.abspath('src'))
    build_standard()
