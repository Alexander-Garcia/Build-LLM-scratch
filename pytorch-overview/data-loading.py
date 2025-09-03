import torch
from torch.utils.data import DataLoader, Dataset

# pytorch requires class labels start with label 0
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


# create a custom dataset class
class ToyDataset(Dataset):
    # 3 main components of custom Dataset class are __init__, __getitem__ and __len__ method.
    # by subclassing we declare an interface contract to implement the methods below
    def __init__(self, X, y):
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        return self.labels.shape[0]


train_ds = ToyDataset(X_train, y_train)
test_ds = ToyDataset(X_test, y_test)

# prints [-1.2, 3.1], 0
print(train_ds[0])
# uses the __len__ which returns the number of samples (y)
print(len(train_ds))

torch.manual_seed(123)

# when setting num_workers=0 data loading will be done in main process and not in separate worker processes
# can lead to significant slowdowns during model training when training larger ntworks
# crucial for parallelizing so the GPU is not idle while CPU processes data
# see fig A.11 for a neat graphic
train_loader = DataLoader(dataset=train_ds, batch_size=2, shuffle=True, num_workers=0)

test_laoder = DataLoader(dataset=test_ds, batch_size=2, shuffle=False, num_workers=0)


for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}: ", x, y)
