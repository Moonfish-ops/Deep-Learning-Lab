"""教材 3.6：softmax 回归从零开始实现。"""

import argparse
from pathlib import Path

import torch
import torchvision
from torch.utils import data
from torchvision import transforms


DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def load_data_fashion_mnist(batch_size):
    transform = transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(
        DATA_ROOT, train=True, transform=transform, download=True
    )
    test_set = torchvision.datasets.FashionMNIST(
        DATA_ROOT, train=False, transform=transform, download=True
    )
    return (
        data.DataLoader(train_set, batch_size, shuffle=True, num_workers=0),
        data.DataLoader(test_set, batch_size, shuffle=False, num_workers=0),
    )


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y])


def accuracy(y_hat, y):
    if y_hat.ndim > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(dim=1)
    return float((y_hat.type(y.dtype) == y).type(torch.float32).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def __getitem__(self, index):
        return self.data[index]


def evaluate_accuracy(net, data_iter):
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train_epoch(net, train_iter, loss, updater):
    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        batch_loss = loss(y_hat, y)
        batch_loss.sum().backward()
        updater(X.shape[0])
        metric.add(batch_loss.detach().sum(), accuracy(y_hat, y), y.numel())
    return metric[0] / metric[2], metric[1] / metric[2]


def main(num_epochs):
    torch.manual_seed(42)
    train_iter, test_iter = load_data_fashion_mnist(batch_size=256)

    W = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
    b = torch.zeros(10, requires_grad=True)

    def net(X):
        return softmax(X.reshape((-1, W.shape[0])) @ W + b)

    def updater(batch_size):
        sgd([W, b], lr=0.1, batch_size=batch_size)

    print("训练前测试准确率:", f"{evaluate_accuracy(net, test_iter):.3f}")
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(
            net, train_iter, cross_entropy, updater
        )
        test_acc = evaluate_accuracy(net, test_iter)
        print(
            f"epoch {epoch + 1:2d}: loss {train_loss:.4f}, "
            f"train acc {train_acc:.3f}, test acc {test_acc:.3f}"
        )

    X, y = next(iter(test_iter))
    pred = net(X).argmax(dim=1)
    print("前 10 个真实标签:", y[:10].tolist())
    print("前 10 个预测标签:", pred[:10].tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    main(args.epochs)
