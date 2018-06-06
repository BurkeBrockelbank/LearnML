# # In this python file, we are going to try to train a virtual machine to replicate the XOR function. The simplest version I have been able to make is a single hidden layer with two nodes feeding into a single output unit. All units are using ReLu.

# import numpy as np
import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Create the net
# class XORNet(nn.Module):
#     def __init__(self):
#         super(XORNet, self).__init__()
#         self.fc1 = nn.Linear(2,2)
#         self.fc2 = nn.Linear(2,1)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         return x

# # Create the training data
# # a | b | aXORb
# #---+---+---
# # 0 | 0 | 0
# # 0 | 1 | 1
# # 1 | 0 | 1
# # 1 | 1 | 0
data = torch.tensor([[0., 0, 0],
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]])
x = data[:,0:2]
y = data[:,[2]]

# # Construct the model as defined in the Net class
# net = XORNet()

# # Construct the loss function and an optimizer
# criterion = nn.MSELoss(size_average=False)
# optimizer = torch.optim.SGD(net.parameters(), lr=1E-4)
# for t in range(10):
#     # Forward pass
#     print('type(inputData) =', type(inputData))
#     print('type(torch.randn(4, 2)) =', type(torch.randn(4, 2)))
#     y = net(inputData)
#     #Loss
#     loss = criterion(y,targetData)

#     #Zero gradients, perform a backward pass, update weights
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

# -*- coding: utf-8 -*-


class XORNet(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(XORNet, self).__init__()
        self.linear1 = torch.nn.Linear(2, 2)
        self.linear2 = torch.nn.Linear(2, 1)

    def forward(self, x):
        """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
        """
        r = torch.nn.functional.leaky_relu
        h = r(self.linear1(x))
        y_pred = r(self.linear2(h))
        return y_pred

# Construct our model by instantiating the class defined above
model = XORNet()
# Initialize the weights to an intelligent guess
model.linear1.weight.data = torch.tensor([[0.9,-1.05],[-1.,1.]])
model.linear1.bias.data = torch.tensor([-0.09,0.2])
model.linear2.weight.data = torch.tensor([[1.03,1.12]])
model.linear2.bias.data = torch.tensor([0.])

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
for t in range(2000):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x)

    # Compute and print loss
    loss = criterion(y_pred, y)
    if t%100 == 99: print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(list(model.parameters()))
v3 = torch.tensor([[0., 1.],])
yPred = model(x)
print(x)
print(yPred)
print(y)
print('loss', criterion(yPred, y))