import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")


text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)

integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

strings = tokenizer.decode(integers)
print(strings)

# exercise test on unknown made up words like
test_string = "Akwirw ier"
test_int = tokenizer.encode(test_string)
print(test_int)

test_strings = tokenizer.decode(test_int)
print(test_strings)
