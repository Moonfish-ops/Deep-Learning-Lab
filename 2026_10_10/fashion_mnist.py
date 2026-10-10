import os
from pathlib import Path

# 避免 Anaconda 的 MKL 与 PyTorch 重复加载 OpenMP 运行库。
# 必须在导入数值计算库前设置。
os.environ["MKL_THREADING_LAYER"] = "SEQUENTIAL"

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data_dir = r"D:\Lab\Deep_Learning_Lab\data"

transform = transforms.ToTensor()

train_data = datasets.FashionMNIST(
    root=data_dir,
    train=True,
    transform=transform,
    download=False
)

test_data = datasets.FashionMNIST(
    root=data_dir,
    train=False,
    transform=transform,
    download=False
)

# 每次取 256 张图片进行训练
batch_size = 256

train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True
)
test_loader = DataLoader(
    test_data, batch_size=batch_size, shuffle=False
)

print("训练集大小：", len(train_data))
print("测试集大小：", len(test_data))

device = torch.device("cpu")
epochs = 10
learning_rate = 0.001

results = {}


# 对比不同激活函数和 Dropout 

experiments = [
    ("ReLU", nn.ReLU(), 0.0),
    ("Tanh", nn.Tanh(), 0.0),
    ("Sigmoid", nn.Sigmoid(), 0.0),
    ("ReLU+Dropout", nn.ReLU(), 0.3)
]

for name, activation, dropout in experiments:

    print("\n开始实验：", name)

    # 使用相同的随机种子
    torch.manual_seed(42)

    # 创建双隐藏层多层感知机
    model = nn.Sequential(
        nn.Flatten(),                  # 28×28 -> 784
        nn.Linear(784, 256),           # 第一隐藏层
        activation,
        nn.Dropout(dropout),
        nn.Linear(256, 128),           # 第二隐藏层
        type(activation)(),
        nn.Dropout(dropout),
        nn.Linear(128, 10)             # 输出 10 个类别
    ).to(device)

    # 交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()

    # Adam 优化器
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate
    )

    # 记录每轮训练结果
    train_losses = []
    train_accs = []
    test_accs = []


    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        # 遍历训练集
        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            # 前向传播
            output = model(X)

            # 计算损失
            loss = loss_fn(output, y)

            # 梯度清零
            optimizer.zero_grad()

            # 反向传播
            loss.backward()

            # 更新参数
            optimizer.step()

            # 统计训练损失和正确数量
            total_loss += loss.item() * y.size(0)
            correct += (output.argmax(dim=1) == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        #  测试模型 

        model.eval()

        test_correct = 0
        test_total = 0

        # 测试时不需要计算梯度
        with torch.no_grad():

            for X, y in test_loader:

                X = X.to(device)
                y = y.to(device)

                output = model(X)

                test_correct += (
                    output.argmax(dim=1) == y
                ).sum().item()

                test_total += y.size(0)

        test_acc = test_correct / test_total

        # 保存本轮结果
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(
            f"Epoch {epoch + 1}/{epochs}  "
            f"Loss: {train_loss:.4f}  "
            f"Train Acc: {train_acc:.4f}  "
            f"Test Acc: {test_acc:.4f}"
        )

    # 保存当前实验结果
    results[name] = (train_losses, train_accs, test_accs)



x = range(1, epochs + 1)

plt.figure(figsize=(15, 4))

# 第一张：训练损失
plt.subplot(1, 3, 1)

for name in results:
    plt.plot(x, results[name][0], label=name)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.grid(True)

# 第二张：训练准确率
plt.subplot(1, 3, 2)

for name in results:
    plt.plot(x, results[name][1], label=name)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy")
plt.legend()
plt.grid(True)

# 第三张：测试准确率
plt.subplot(1, 3, 3)

for name in results:
    plt.plot(x, results[name][2], label=name)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Test Accuracy")
plt.legend()
plt.grid(True)

plt.tight_layout()

result_path = Path(__file__).resolve().with_name("fashion_mnist_result.png")
plt.savefig(result_path, dpi=300)
plt.close()
print(f"结果图已保存：{result_path}")


for name in results:
    print(
        f"{name}: "
        f"训练准确率 = {results[name][1][-1]:.4f}, "
        f"测试准确率 = {results[name][2][-1]:.4f}"
    )

