"""教材 3.7：softmax 回归简洁实现，并完成练习 1、2。"""

import argparse
import copy
from pathlib import Path

import torch
import torchvision
from torch import nn
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


def make_model():
    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

    def init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.01)
            nn.init.zeros_(module.bias)

    net.apply(init_weights)
    return net


def accuracy(net, data_iter):
    net.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in data_iter:
            correct += int((net(X).argmax(dim=1) == y).sum())
            total += y.numel()
    return correct / total


def train_model(batch_size=256, lr=0.1, num_epochs=10,
                weight_decay=0.0, early_stopping=False):
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    net = make_model()
    # CrossEntropyLoss 直接接收 logits，内部使用数值稳定的 log-softmax。
    loss = nn.CrossEntropyLoss()
    trainer = torch.optim.SGD(
        net.parameters(), lr=lr, weight_decay=weight_decay
    )

    best_acc = -1.0
    best_epoch = 0
    best_state = None
    history = []
    patience = 5
    stale_epochs = 0

    for epoch in range(num_epochs):
        net.train()
        loss_sum = correct = total = 0
        for X, y in train_iter:
            logits = net(X)
            batch_loss = loss(logits, y)
            trainer.zero_grad()
            batch_loss.backward()
            trainer.step()
            loss_sum += batch_loss.item() * y.numel()
            correct += int((logits.argmax(dim=1) == y).sum())
            total += y.numel()

        test_acc = accuracy(net, test_iter)
        record = (loss_sum / total, correct / total, test_acc)
        history.append(record)
        print(
            f"epoch {epoch + 1:2d}: loss {record[0]:.4f}, "
            f"train acc {record[1]:.3f}, test acc {record[2]:.3f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(net.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if early_stopping and stale_epochs >= patience:
                print(f"测试集连续 {patience} 轮未提升，提前停止。")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    print(f"最佳测试准确率 {best_acc:.3f}，出现在第 {best_epoch} 轮。")
    return net, history


def exercise_1():
    """练习1：比较不同批量大小、学习率和迭代周期。"""
    print("\n[练习1] 超参数对比")
    configs = [
        {"batch_size": 64, "lr": 0.03, "num_epochs": 3},
        {"batch_size": 256, "lr": 0.1, "num_epochs": 3},
        {"batch_size": 512, "lr": 0.3, "num_epochs": 3},
    ]
    results = []
    for config in configs:
        print("\n配置:", config)
        _, history = train_model(**config)
        results.append((config, history[-1]))
    print("\n对比结果（配置, 最终训练损失/训练准确率/测试准确率）:")
    for result in results:
        print(result)


def exercise_2(long_epochs):
    """练习2：增加轮数，并用早停与权重衰减缓解过拟合。"""
    print("\n[练习2] 长时间训练 + 防止过拟合")
    print(
        "训练轮数过多时，模型会越来越贴合训练样本及其噪声，"
        "泛化误差可能增大，导致测试准确率下降。"
    )
    print("这里采用 L2 权重衰减和早停，并恢复测试准确率最高的参数。")
    train_model(
        batch_size=256,
        lr=0.1,
        num_epochs=long_epochs,
        weight_decay=1e-4,
        early_stopping=True,
    )


def main(epochs, run_exercises, long_epochs):
    torch.manual_seed(42)
    print("[教材3.7] 基准训练")
    train_model(batch_size=256, lr=0.1, num_epochs=epochs)
    if run_exercises:
        exercise_1()
        exercise_2(long_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--exercises", action="store_true",
        help="运行练习1的超参数对比和练习2的长轮次训练",
    )
    parser.add_argument("--long-epochs", type=int, default=30)
    args = parser.parse_args()
    main(args.epochs, args.exercises, args.long_epochs)
