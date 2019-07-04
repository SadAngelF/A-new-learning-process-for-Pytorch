'''
# 一个简单的pytorch测试和使用示范

# by SadAngel

'''
from __future__ import print_function
import torch

# 创建未初始化的矩阵：

X1 = torch.empty(5, 3)
print(X1)

# 创建一个随机初始化矩阵：

X2 = torch.rand(5, 3)
print(X2)

# 创建一个0填充的矩阵，类型为long：

X3 = torch.zeros(5, 3, dtype=torch.long)
print(X3)

# 创建tensor并使用现有数据初始化：
X4 = torch.tensor([5.5, 3])
print(X4)

# 根据现有tensor创建tensor:

X5 = X4.new_ones(5, 3, dtype=torch.double)
print(X5)

X6 = torch.randn_like(X5, dtype=torch.float)
print(X6)

# 获取size：

print(X6.size())  #返回值是tuple类型，支持各种操作


# 这部分之后为一些tensor的操作：

# 加法：

X = torch.rand(5, 3)
Y = torch.rand(5, 3)
print(X, Y, X+Y)

# 

print(torch.add(X, Y))

# 提供输出tensor作为参数：

result = torch.empty(5, 3)
torch.add(X, Y, out=result)
print(result)

# 替换:

Y.add_(X)
print(Y)

# 索引与numpy相同：

print(Y[:,1])

# torch.view可以改变张量维度和大小，与np.reshape类似

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  #  size -1 从其他维度推断
print(x.size(), y.size(), z.size())

# 如果你有只有一个元素的张量，使用.item()来得到Python数据类型的数值

x = torch.randn(1)
print(x)
print(x.item())

# 使用.to方法将tensor移动到其他设备中，例如CUDA
# is_available 函数判断是否有cuda可以使用
# ``torch.device``将张量移动到指定的设备中
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 直接从GPU创建张量
    x = x.to(device)                       # 或者直接使用``.to("cuda")``将张量移动到cuda中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` 也会对变量的类型做更改
