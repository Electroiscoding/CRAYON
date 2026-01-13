"""Quick verification of the updated vocabulary with code tokens."""

from crayon import CrayonVocab

# Load vocabulary
v = CrayonVocab.from_json('trained_vocab.json')
print(f"Vocabulary Size: {len(v):,} tokens")
print(f"C-Extension: {'Enabled' if v._c_ext_available else 'Disabled'}")

# Test code samples from multiple languages
test_cases = [
    ("Python", "def fibonacci(n: int) -> int:\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"),
    ("JavaScript", "const fetchData = async (url) => { const res = await fetch(url); return res.json(); }"),
    ("TypeScript", "interface User { id: number; name: string; email: string; }"),
    ("Java", 'public static void main(String[] args) { System.out.println("Hello World"); }'),
    ("C++", "#include <iostream>\nint main() { std::cout << \"Hello\" << std::endl; return 0; }"),
    ("Rust", 'fn main() { let x: i32 = 42; println!("Value: {}", x); }'),
    ("Go", 'func main() { fmt.Println("Hello, World!") }'),
    ("NumPy", "import numpy as np\ndf = pd.DataFrame(data)"),
]

print("\n" + "=" * 50)
print("Verification Tests")
print("=" * 50)

for lang, code in test_cases:
    tokens = v.tokenize(code)
    decoded = v.decode(tokens)
    match = "[OK]" if decoded == code else "[FAIL]"
    
    display = code[:45] + "..." if len(code) > 45 else code
    display = display.replace('\n', '\\n')
    print(f"\n[{lang}] {match}")
    print(f"  Input:  '{display}'")
    print(f"  Tokens: {len(tokens)}")

print("\n" + "=" * 50)
print("Sample Code Tokens (IDs 50000+)")
print("=" * 50)

# Show some new code tokens (starting after the original 50k)
print("\nNew code tokens (sample):")
for i in range(50000, min(50030, len(v))):
    token = v.id_to_token[i]
    display = repr(token) if len(repr(token)) < 30 else repr(token[:25] + "...")
    print(f"  ID {i}: {display}")

print(f"\nTotal vocabulary: {len(v):,} tokens")
