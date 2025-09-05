import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data_loading import ToyDataset
from multilayer import NeuralNetwork

torch.manual_seed(123)
X_train = torch.tensor(
    [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
)
print("X_train shape: ", X_train.shape)
y_train = torch.tensor([0, 0, 0, 1, 1])


X_test = torch.tensor(
    [
        [-0.8, 2.8],
        [2.6, -1.6],
    ]
)
y_test = torch.tensor([0, 1])

train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0)

test_laoder = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)
model = NeuralNetwork(num_inputs=2, num_outputs=2)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):
        # refers to the unscaled output of a model
        logits = model(features)

        loss = F.cross_entropy(logits, labels)

        # prevent undesired gradient accumulation
        optimizer.zero_grad()
        loss.backward()
        # use gradients to update model params and min loss
        optimizer.step()

        print(
            f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
            f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
            f" | Train Loss: {loss:.2f}"
        )

    model.eval()

model.eval()
with torch.no_grad():
    outputs = model(X_train)
print(outputs)


# to obtain class membership
torch.set_printoptions(sci_mode=False)
prob = torch.softmax(outputs, dim=1)
print(prob)


# saving a model
torch.save(model.state_dict(), "model.pth")

# loading it
# this line not strictly necessary if executed in same session where saving a model
# just illustrates we need an instance of the model in memory to apply saved params
# num_in and out needs to match original saved model exactly
model = NeuralNetwork(2, 2)
model.load_state_dict(torch.load("model.pth"))
