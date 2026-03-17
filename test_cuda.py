from crayon import CrayonVocab
tokenizer = CrayonVocab(device="cuda")
tokenizer.load_profile("standard")
tokens = tokenizer.tokenize("someunknownPlace")
print(tokens)
