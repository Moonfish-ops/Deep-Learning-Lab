import random

import torch


def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


def data_iter(batch_size, features, labels):
    """随机打乱样本，并按小批量读取数据。"""
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i : min(i + batch_size, num_examples)]
        )
        yield features[batch_indices], labels[batch_indices]


def linreg(X, w, b):
    """线性回归模型。"""
    return torch.matmul(X, w) + b


def squared_loss(y_hat, y):
    """平方损失。reshape 保证标签与预测值的形状一致。"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def sgd(params, lr, batch_size):
    """小批量随机梯度下降。"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


def train(features, labels, zero_init=False):
    """使用教材中的训练循环拟合线性回归模型。"""
    if zero_init:
        w = torch.zeros((2, 1), requires_grad=True)
    else:
        w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    batch_size = 10
    lr = 0.03
    num_epochs = 3

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            loss = squared_loss(linreg(X, w, b), y)
            loss.sum().backward()
            # 本例 1000 能被 10 整除，所以每批实际大小均为 batch_size。
            sgd([w, b], lr, batch_size)

        with torch.no_grad():
            train_loss = squared_loss(linreg(features, w, b), labels)
            print(f"epoch {epoch + 1}, loss {train_loss.mean().item():.6f}")

    return w, b


def exercise_1(features, labels):
    """1、如果我们将权重初始化为零，会发生什么。算法仍然有效吗？"""
    print("\n 权重初始化为零")
    w, b = train(features, labels, zero_init=True)
    print("零初始化后 w:", w.detach().reshape(-1).tolist())
    print("零初始化后 b:", b.detach().item())


def exercise_5():
    """5、不 reshape 会发生什么。"""
    print("\n reshape 的作用")
    y_hat = torch.tensor([[1.0], [2.0], [3.0]])  # 形状 (3, 1)
    y = torch.tensor([1.1, 1.9, 3.2])            # 形状 (3,)
    wrong_loss = (y_hat - y) ** 2 / 2
    correct_loss = squared_loss(y_hat, y)
    print("y_hat 形状:", tuple(y_hat.shape), "y 形状:", tuple(y.shape))
    print("未 reshape 的损失形状:", tuple(wrong_loss.shape))
    print("reshape 后的损失形状:", tuple(correct_loss.shape))
    print("reshape 后逐样本损失:", correct_loss.reshape(-1).tolist())


def exercise_7():
    """7、样本数不能被批量大小整除"""
    print("\n 样本数不能被批量大小整除")
    demo_features = torch.arange(46, dtype=torch.float32).reshape(23, 2)
    demo_labels = torch.arange(23, dtype=torch.float32).reshape(-1, 1)
    batch_sizes = [len(X) for X, _ in data_iter(10, demo_features, demo_labels)]
    print("23 个样本、batch_size=10，各批大小:", batch_sizes)


def main():
    # 固定随机种子，便于复现实验结果。
    torch.manual_seed(42)
    random.seed(42)

    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    print("第一个样本特征:", features[0])
    print("第一个样本标签:", labels[0])
    print("\n[教材3.2] 随机初始化训练")
    w, b = train(features, labels)

    print("w 的估计误差:", true_w - w.detach().reshape(true_w.shape))
    print("b 的估计误差:", true_b - b.detach())

    exercise_1(features, labels)
    exercise_5()
    exercise_7()


if __name__ == "__main__":
    main()
