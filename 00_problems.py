import torch
torch.manual_seed(0)
RAND_1 = torch.rand(size=(7, 7))
print(RAND_1)

RAND_2 = torch.rand(size=(1, 7))
print(RAND_2)

print(RAND_1 @ RAND_2.T)


print(RAND_1.min())
print(RAND_2.max())

RAND_3 = torch.rand(size=(1, 1, 1, 10))
print(RAND_3)
print(RAND_3.squeeze())
