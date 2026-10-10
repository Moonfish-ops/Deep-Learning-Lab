import torch
from torch import nn

class WrongSequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        #self.layers = []
        self.layers = nn.ModuleList()

        for module in args:
            self.layers.append(module)

    def forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X


net = WrongSequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

X = torch.rand(2, 20)

print(net(X).shape)

print("模型参数：")
for name, param in net.named_parameters():
    print(name, param.shape)