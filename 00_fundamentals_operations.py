import torch

#Addition
tensor = torch.tensor([1, 2, 3])
print(tensor)
print(tensor + 10)
    #tensor([11, 12, 13])
print(tensor)
    #tensor([1, 2, 3]) -> tensor won't change unless reassigned

#Subtraction
print(tensor - 10)
    #tensor([-9, -8, -7])

#Reassignment
tensor = tensor + 10
print(tensor)
    #tensor([11, 12, 13])

#Multiplication
tensor = torch.tensor([1, 2, 3])
print(torch.multiply(tensor, 10))
    #tensor([10, 20, 30])
print(tensor * tensor)
    #tensor([1, 4, 9)
print(tensor * 10)
    #tensor([10, 20, 30])

#Matrix Multiplication
print(torch.matmul(tensor, tensor))
     #tensor(14)
     #How does it multiply two vectors?
print(tensor @ tensor)
     #tensor(14)

#Transpose
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11], 
                         [9, 12]], dtype=torch.float32)
#torch.matmul(tensor_A, tensor_B) # (this will error)

print(tensor_B)
print(tensor_B.T) #transpose
print(tensor_A @ tensor_B.T)
    #tensor([[ 27.,  30.,  33.],
    #        [ 61.,  68.,  75.],
    #        [ 95., 106., 117.]])
print(torch.matmul(tensor_A, tensor_B.T)) 
    #tensor([[ 27.,  30.,  33.],
    #        [ 61.,  68.,  75.],
    #        [ 95., 106., 117.]])
print(torch.mm(tensor_A, tensor_B.T)) # -> shortcut for matmul
    #tensor([[ 27.,  30.,  33.],
    #        [ 61.,  68.,  75.],
    #        [ 95., 106., 117.]])

#Aggregating Tensors
x = torch.arange(0, 100, 10)
print(x)
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")

print(torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x))

#Positional Min/Max
# Create a tensor
tensor = torch.arange(10, 100, 10)
print(f"Tensor: {tensor}")

# Returns index of max and min values
print(f"Index where max value occurs: {tensor.argmax()}")
print(f"Index where min value occurs: {tensor.argmin()}")

#Changing Data Type
tensor = torch.arange(10., 100., 10.)
print(tensor)
print(tensor.type(torch.float16))

#Reshaping Tensors
x = torch.arange(1., 8.)
print(x, x.shape)
x_reshaped = x.reshape(1, 7)
print(x_reshaped, x_reshaped.shape)

#Changing The view
z = x.view(1, 7)
print(z, z.shape)
z[:, 0] = 5
print(z,x)

#Note, reshape and view are very similar, but reshape doesn't guarantee the resulting tensor will be contiguous 
#Also, view still points to the underlying tensor, and the returned reshaped tensor does not

#Stacking Tensors
x_stacked = torch.stack([x, x, x, x], dim=0) # try changing dim to dim=1 and see what happens
print(x_stacked)

#Squeeing Tensors -> removes all dimensions that are only 1
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}")

# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()
print(f"\nNew tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}")
#[[5., 2., 3., 4., 5., 6., 7.]] -> [5., 2., 3., 4., 5., 6., 7.]


## Add an extra dimension with unsqueeze
print(f"Previous tensor: {x_squeezed}")
print(f"Previous shape: {x_squeezed.shape}")
x_unsqueezed = x_squeezed.unsqueeze(dim=0)
print(f"\nNew tensor: {x_unsqueezed}")
print(f"New shape: {x_unsqueezed.shape}")

#Permuting Tensors
x_original = torch.rand(size=(224, 224, 3))
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0

print(x_original)
print(f"Previous shape: {x_original.shape}")
#Previous shape: torch.Size([224, 224, 3])
print(x_permuted)
print(f"New shape: {x_permuted.shape}")
#New shape: torch.Size([3, 224, 224])

#note, permuted tensor shares a view with the original, and so changing values will change values on the original


