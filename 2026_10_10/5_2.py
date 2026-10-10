import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)

    def forward(self, X):
        return self.out(torch.relu(self.hidden(X)))

net = MLP()

# 访问隐藏层的权重
print("权重形状：", net.hidden.weight.shape)

# 访问隐藏层的偏置
print("偏置形状：", net.hidden.bias.shape)

# 查看权重的实际数值
print("前5个权重：", net.hidden.weight.data[0, :5])

# 查看所有参数的名字和形状
for name, param in net.named_parameters():
    print(name, param.shape)