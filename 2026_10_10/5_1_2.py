import torch
from torch import nn

class ParallelBlock(nn.Module):
    def __init__(self, net1, net2):
        super().__init__()
        self.net1 = net1
        self.net2 = net2

    def forward(self, X):
        Y1 = self.net1(X)
        Y2 = self.net2(X)
        return torch.cat((Y1, Y2), dim=1)


net1 = nn.Linear(20, 10)
net2 = nn.Linear(20, 5)

net = ParallelBlock(net1, net2)

X = torch.randn(2, 20)
Y = net(X)

print("net1 输出：", net1(X).shape)
print("net2 输出：", net2(X).shape)
print("平行块输出：", Y.shape)