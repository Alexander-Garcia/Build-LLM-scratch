import torch


# subclass the torch.nn.Module class to define our own custom network architecture
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        # Sequential not required but makes life easier if we have series of layers we want to execute in specific order
        # doing this in __init__ so we just have to call self.layers instead of calling each layer individually in Networks forward method
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    # describes how input data passes through the network and comes together as computation graph
    def forward(self, x):
        logits = self.layers(x)
        return logits


# instantiate new neural network object:
model = NeuralNetwork(num_inputs=50, num_outputs=3)

# check it out
print(model)

# check total number of trainable params
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable params: ", num_params)


# from model print above can see index position of layers
# access weight parameter matrix of 1st layer
# model weights init with small random numbers to break symmetry during training
# otherwise nodes would be performing same operations and updates during backprop
# could use torch.manual_seed() to make it reproducible
print(model.layers[0].weight)

# shape it
print(model.layers[0].weight.shape)


# see how its used during forward pass
torch.manual_seed(123)
X = torch.rand((1, 50))
# this will automatically execute forward pass
out = model(X)
print(out)


# if we want no backprop for example prediction after training
with torch.no_grad():
    out = model(X)
print("No grad: ", out)


# common practice to code models such that they return outputs of last layer (logits) without passing them ot nonlinear act func
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
print("Softmax: ", out)
