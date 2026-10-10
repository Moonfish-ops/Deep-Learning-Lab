import torch
from torch import nn


class QuadraticLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.W = nn.Parameter(
            torch.randn(in_features, in_features, out_features)
        )

    def forward(self, X):
        Y = torch.einsum('bi,bj,ijk->bk', X, X, self.W)
        return Y


# 4 个输入特征，3 个输出特征
layer = QuadraticLayer(4, 3)

# 2 个样本，每个样本 4 个特征
X = torch.randn(2, 4)

# 前向传播
Y = layer(X)

print("输入形状：", X.shape)
print("权重形状：", layer.W.shape)
print("输出形状：", Y.shape)

loss = Y.square().mean()
loss.backward()

print("权重梯度形状：", layer.W.grad.shape)
