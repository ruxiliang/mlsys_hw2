import os
import time
import numpy as np
from needle.data.data_transforms import FlattenMnist
import needle.nn as nn
import needle as ndl
import sys

sys.path.append("../python")

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1) -> nn.Module:
    # BEGIN YOUR SOLUTION
    return nn.Sequential(
            nn.Residual(
                nn.Sequential(
                    nn.Linear(in_features=dim, out_features=hidden_dim),
                    norm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(p=drop_prob),
                    nn.Linear(in_features=hidden_dim, out_features=dim),
                    norm(dim)
                )), 
            nn.ReLU()
        )

    # END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    # BEGIN YOUR SOLUTION
    return nn.Sequential(
        nn.Linear(in_features=dim, out_features=hidden_dim),
        nn.ReLU(),
        *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim // 2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
        nn.Linear(in_features=hidden_dim, out_features=num_classes)
    )
    # END YOUR SOLUTION



def epoch(dataloader, model: nn.Module, opt=None):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    if opt is None:
        model.eval()
    else:
        model.train()
    all_losses = []
    all_error = 0
    dataset_len = len(dataloader.dataset)
    for X, y in dataloader:
        logits: ndl.Tensor = model(X)
        loss = nn.SoftmaxLoss()
        l = loss(logits, y)
        prediction = logits.numpy().argmax(axis=1)
        all_error += np.sum(prediction != y.numpy())
        if opt is not None:
            l.backward()
            opt.step()
        all_losses.append(l.numpy())
    return all_error / dataset_len, np.array(all_losses).mean()
    # END YOUR SOLUTION

def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    # BEGIN YOUR SOLUTION
    train_dataset = ndl.data.mnist_dataset.MNISTDataset(os.path.join(data_dir,'train-images-idx3-ubyte.gz'), os.path.join(data_dir,'train-labels-idx1-ubyte.gz'), transforms=[FlattenMnist()])
    test_dataset = ndl.data.mnist_dataset.MNISTDataset(os.path.join(data_dir,'t10k-images-idx3-ubyte.gz'), os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz'), transforms=[FlattenMnist()])
    training_accuracy, training_loss, test_accuracy, test_loss = 0, 0, 0, 0
    model = MLPResNet(dim=784, hidden_dim=hidden_dim)
    for _ in range(epochs):    
        train_dataloader = ndl.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = ndl.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        training_err, training_loss = epoch(train_dataloader, model, opt=optimizer(model.parameters(), lr=lr, weight_decay=weight_decay))
        test_err, test_loss = epoch(test_dataloader, model)
        training_accuracy, test_accuracy = 1 - training_err, 1 - test_err
    return training_err, training_loss, test_err, test_loss
    # END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
