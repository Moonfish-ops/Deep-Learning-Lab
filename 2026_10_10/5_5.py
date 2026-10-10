
import torch
from torch import nn

# 1. 创建旧网络
old_net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 2. 保存旧网络前两层的状态
torch.save(old_net[:2].state_dict(), "first_two.pth")

# 3. 创建新网络
new_net = nn.Sequential(
    nn.Linear(20, 256),
    nn.ReLU(),
    nn.Linear(256, 5)
)

# 4. 加载旧网络前两层的参数
params = torch.load("first_two.pth", weights_only=True)
new_net[:2].load_state_dict(params)

# 5. 验证前两层的权重是否一致
print("权重是否一致：",
      torch.allclose(old_net[0].weight, new_net[0].weight))

# 6. 验证新网络能否正常运行
X = torch.randn(2, 20)
Y = new_net(X)

print("新网络输出形状：", Y.shape)
