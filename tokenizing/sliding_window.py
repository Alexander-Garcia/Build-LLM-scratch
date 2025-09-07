import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))


# removing first 50 set of tokens for demonstrating
enc_sample = enc_text[50:]

# One of the easiest and most intuitive ways to create the input–target pairs for the next-word prediction task is to create two variables
# x and y, where x contains the input tokens and y contains the targets, which are the inputs shifted by 1:
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1 : context_size + 1]
print(f"x: {x}")
print(f"y:      {y}")

# Everything left of the arrow would be input LLM is to receive
# to right is what the LLM should predict
for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)


for i in range(1, context_size + 1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
