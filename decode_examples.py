from crayon import CrayonVocab

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")

text = "Hello, world!"
tokens = vocab.tokenize(text)
print(tokens)
decode=vocab.decode(tokens)
print(decode)