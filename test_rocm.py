from crayon import CrayonVocab
try:
    tokenizer = CrayonVocab(device="rocm")
    tokenizer.load_profile("standard")
    tokens = tokenizer.tokenize("someunknownPlace")
    print(tokens)
except Exception as e:
    print(e)
