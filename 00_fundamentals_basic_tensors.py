import torch

#Scalar
scalar = torch.tensor(7)
print(scalar)
print(scalar.ndim)
print(scalar.item()) #only works with 1 element tensors

#Vector
vector = torch.tensor([7, 7])
print(vector)
print(vector.ndim)
print(vector.shape)

#Matrix
MATRIX = torch.tensor([[7, 8],
                       [9, 10]])
print(MATRIX)
print(MATRIX.ndim)
print(MATRIX.shape)

#Tensor
TENSOR = torch.tensor([[[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]]])
print(TENSOR)
print(TENSOR.ndim)
print(TENSOR.shape)

#Random Tensor
random_tensor = torch.rand(size=(3, 4))
print(random_tensor)
print(random_tensor.dtype)

#Zeros Tensor
zeros = torch.zeros(size=(3, 4))
print(zeros)
print(zeros.dtype)

#Ones Tensor
ones = torch.ones(size=(3, 4))
print(ones)
print(ones.dtype)

#Zeroes like -> same shape as input
zeros_like = torch.zeros_like(TENSOR)
print(zeros_like)
print(zeros.dtype)

#Ones like
ones_like = torch.ones_like(TENSOR)
print(ones_like)
print(ones.dtype)

#16 Float Tensor
float_16_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=torch.float16) # torch.half would also work

print(float_16_tensor)
print(float_16_tensor.dtype)
