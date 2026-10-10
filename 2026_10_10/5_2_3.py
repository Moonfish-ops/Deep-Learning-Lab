import torch
from torch import nn

shared = nn.Linear(20, 20)

net = nn.Sequential(
    nn.Linear(20, 20),
    nn.ReLU(),
    shared,
    nn.ReLU(),
    shared
)


# 统计参数总量
print("参数总量：", sum(p.numel() for p in net.parameters()))

# 模拟一次训练
X = torch.randn(4, 20)
target = torch.randn(4, 20)

optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

optimizer.zero_grad()

Y = net(X)
loss = loss_fn(Y, target)

loss.backward()

print("共享层梯度形状：", shared.weight.grad.shape)
print("共享层梯度范数：", shared.weight.grad.norm().item())

optimizer.step()

print("更新后仍然共享：", net[2].weight is net[4].weight)