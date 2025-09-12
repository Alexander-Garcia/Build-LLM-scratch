import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

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


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return dataloader


dataloader = create_dataloader_v1(
    raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)
data_iter = iter(dataloader)
# contains two tensors: input token IDs and target token IDs
# max_length = 4 means each of the two tensors contains 4 token IDs
first_batch = next(data_iter)
print(first_batch)

# shows each tensor in the batch has shifted by 1 - the stride
# stride = how much we shift the sliding window
second_batch = next(data_iter)
print(second_batch)
# small batch sizes require less memory but lead to more noisy model updates.
# batch size is a tradeoff and a hyperparameter to experiment with when training LLMs


# batch size refers to amount of rows in 2d tensor
# check data loader to sample with batch size > 1
# this has 8 rows, each of length 4 and shifts by 4
# With this there is no more overlap between rows like
# [[the, quick, fox]
# [quick, fox, rocks]]
# would now be
# [[the, quick, fox]
# [rocks, the, guitar]]
dataloader = create_dataloader_v1(
    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs: ", inputs)
print("Targets: ", targets)
