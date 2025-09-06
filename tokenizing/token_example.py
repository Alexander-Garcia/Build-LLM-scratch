import re

from simple_tokenizer import SimpleTokenizerV1

with open("the-verdict.txt", "r", encoding="utf-8") as file:
    raw_text = file.read()
print("Total number of characters: ", len(raw_text))
print(raw_text[:99])

# just a small excursion for illustration purposes
text = "Hello, world. This, is a test."
results = re.split(r"(\s)", text)
print(results)


# refined a little more punctuation
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)


# use basic tokenizer on verdict text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))
print(preprocessed[:30])


# Next convert tokens from a python string to integer rep to produce token IDs
# conversion is intermediate step before converint IDs into embedding vectors
# to map prev generated tokens into token ID build a "vocabulary" first
# this defines how we map each unique word and special char to unique integer
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)

# create the vocab from sorted words
vocab = {token: integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50:
        break


tokenizer = SimpleTokenizerV1(vocab)
text = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)

print(tokenizer.decode(ids))


# what happens if it finds a token not in vocab?
# how about splitting text
all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token: integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))

# double check the new ones are there
for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)


text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."
text = " <|endoftext|> ".join([text1, text2])
print(text)


tokenizer = SimpleTokenizerV1(vocab)
print(tokenizer.decode(tokenizer.encode(text)))
# depending on LLM some researchers also consider additional special tokens such
# [ BOS ] (beg of seq) - start of text. Tells LLM where piece of content begins
# [ EOS ] (end of seq) - useful when concat mult unrelated texts
# [ PAD ] (padding) - when training with batch sizers larger than one batch might contain varying len.
# to ensure all have same len shorter texts are extended or "padded" using this token
