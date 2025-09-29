# Build-LLM-scratch
Notes and code from reading Build A Large Language Model by S. Raschka

## Appendix A - PyTorch refresher
[PyTorch Overview](./pytorch-overview) covers basic tensor creation and operations as well as autograd and subclassing.

## Chapter 1 - Understanding large language models
- general process of creating LLM includes pretraining and fine-tuning
    - "pre" in "pretraining" refers to the initial phase where a model is trained on large, diverse dataset
        - in this stage LLMs use self-supervised learning where model generates its own labels from input data
    - this servers as foundational resource that can be further refined through fine-tuning
- **instruction fine-tuning** the labeled dataset consists of instruction and answer pairs
- **classification fine-tuning** labeled dataset consists of texts and associated class labels - emails associated with "spam" and "not spam" labels
- decoder-style models like GPT generate text by predicting text one word at a time
    - considered **autoregressive**
    - autoregressive models incorporate their previous outputs as inputs for future predictions
- **emergent behavior** model's ability to perform tasks it wasn't explicitly trained to perform

## Chapter 2 - Working with text data
[tokenizing](./tokenizing)
- **embedding** concept of converting data into vector format
    - mapping from discrete objects, such as words, images, or even entire documents, to points in continuous vector space
- **vocabulary** - mapping of unique tokens (words or characters) to unique integer IDs
- Refrain from making all text lowercase - capitalization helps LLMs distinguish proper nouns, common nouns, understand sentence struct and learn to generate text with proper capitalization
- Keeping whitespaces can be useful if training models that are sensitive to exact structure (like Python code)
- Tokenization breaks down input text into individual tokens
    - each unique token added to vocab in alphabetical order
    - this mapping maps tokens to token IDs

| Unique Token | Token ID |
|--------------|----------|
| Brown        | 0        |
| dog          | 1        |
| fox          | 2        |
| jumps        | 3        |
| lazy         | 4        |


- Special tokens like <| endoftext |> and <| unk |> help separate text and account for unknown tokens
    - there can be others in diff format like [ BOS ], [ EOS ], and more depending on the LLM
- However, more sophisticated tokenization schemes like BPE can handle unknown words
    - see tiktoken (implemented in tokenizing dir)
- Algorithm in BPE breaks down words that aren't in its predefined vocab into smaller subword units or even indv characters, enabling out-of-vocabulary words
- 2 broad categories of position-aware embeddings
    - **relative positional embeddings**
        - emphasis on distance between tokens
        - model learns relationship in terms of "how far apart" rather than "at which exact position"
        - advantage is model can generalize better to sequences of varying lengths, even if it hasn't see such during training
    - **absolute positional embeddings**
        - directly associated with specific positions in a sequence
        - each position in input sequence, unique embedding is added to token's embedding to convey exact location
- the choice between the two often depends on specific application and nature of data being processed

