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
