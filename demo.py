"""
XERV Crayon Demo Script.

Demonstrates the core functionality including:
1. Basic tokenization
2. Pipeline processing
3. C-extension status check
"""

import time
from crayon import CrayonVocab, PipelineTokenizer, check_c_extension, check_resources


def main():
    print("=" * 60)
    print("XERV Crayon Tokenizer Demo")
    print("=" * 60)
    
    # 1. Check C-extension status
    print("\n[1] System Status")
    print(f"    C-Extension: {'[OK] Enabled (SIMD)' if check_c_extension() else '[--] Disabled (Python)'}")
    
    resources = check_resources()
    print(f"    HuggingFace: {'[OK] Available' if resources.get('huggingface_available') else '[--] Not installed'}")
    print(f"    Requests: {'[OK] Available' if resources.get('requests_available') else '[--] Not installed'}")
    
    # 2. Initialize Vocabulary
    print("\n[2] Initializing Vocabulary...")
    tokens = [
        "<PAD>", "<UNK>", "<BOS>", "<EOS>",
        "hello", "world", "production", "grade", 
        "tokenizer", "xerv", "crayon", " ", "!", ".",
        "the", "a", "is", "this", "test"
    ]
    vocab = CrayonVocab(tokens)
    print(f"    Vocabulary size: {len(vocab)} tokens")
    print(f"    C-Trie built: {vocab._c_ext_available}")
    
    # 3. Basic Tokenization
    text = "hello world this is a test!"
    print(f"\n[3] Tokenizing: '{text}'")
    
    start = time.perf_counter()
    ids = vocab.tokenize(text)
    elapsed = (time.perf_counter() - start) * 1000
    
    print(f"    Token IDs: {ids}")
    print(f"    Decoded: {vocab.decode(ids)}")
    print(f"    Time: {elapsed:.3f}ms")
    
    # 4. Throughput Test
    print("\n[4] Throughput Test (1M iterations)...")
    test_text = "hello world " * 100
    iterations = 10000
    
    start = time.perf_counter()
    for _ in range(iterations):
        _ = vocab.tokenize(test_text)
    elapsed = time.perf_counter() - start
    
    tokens_per_iter = len(vocab.tokenize(test_text))
    total_tokens = tokens_per_iter * iterations
    throughput = total_tokens / elapsed
    
    print(f"    Tokens processed: {total_tokens:,}")
    print(f"    Time: {elapsed:.3f}s")
    print(f"    Throughput: {throughput:,.0f} tokens/sec")
    
    # 5. Pipeline Demo
    print("\n[5] Pipeline Processing...")
    pipeline = PipelineTokenizer(vocab)
    pipeline.start_pipeline()
    
    docs = [
        ("doc_1", "hello world"),
        ("doc_2", "this is crayon"),
        ("doc_3", "production grade tokenizer"),
    ]
    
    for doc_id, text in docs:
        pipeline.submit_text(doc_id, text)
    
    for _ in range(len(docs)):
        result = pipeline.get_result(timeout=5.0)
        print(f"    {result['id']}: {result['input_ids']} (length: {result['length']})")
    
    pipeline.stop_pipeline()
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
