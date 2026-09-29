"""教材 3.3：使用 PyTorch 高级 API 简洁实现线性回归。"""

import torch
from torch import nn
from torch.utils import data


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = X @ w + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def main():
    torch.manual_seed(42)
    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    batch_size = 10
    data_iter = load_array((features, labels), batch_size)
    X, y = next(iter(data_iter))
    print("首个小批量形状:", X.shape, y.shape)

    # 一个展平的全连接层：2 个输入，1 个输出。
    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)

    for epoch in range(3):
        for X, y in data_iter:
            batch_loss = loss(net(X), y)
            trainer.zero_grad()
            batch_loss.backward()
            trainer.step()
        epoch_loss = loss(net(features), labels)
        print(f"epoch {epoch + 1}, loss {epoch_loss.item():.6f}")

    w = net[0].weight.data
    b = net[0].bias.data
    print("学到的 w:", w.reshape(-1).tolist())
    print("学到的 b:", b.item())
    print("w 的估计误差:", true_w - w.reshape(true_w.shape))
    print("b 的估计误差:", true_b - b)
    # 如需访问梯度，可在 trainer.zero_grad() 前查看 net[0].weight.grad。


if __name__ == "__main__":
    main()
