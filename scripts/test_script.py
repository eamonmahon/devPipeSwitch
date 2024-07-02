import torch

x = torch.randn(10, 10).cuda()
print(x)
y = torch.randn(10, 10, device='cuda')
print(y)
