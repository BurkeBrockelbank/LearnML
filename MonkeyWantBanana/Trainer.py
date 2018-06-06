# This impliments some supervised training for the monkey.
#

from __future__ import print_function
from __future__ import division
import Roomgen
import Monkey
import Grid
import Exceptions
from random import randrange
import torch

def generateTrainingDataSporadic(N, filePath,abstractMap):
    # This is a creator for generating data from the map
    # The monkey starts at a random location each tick
    # Create a monkey
    testMonkey = Monkey.Monkey()
    # Create the grid
    g = Grid.Grid([testMonkey], [(3,3)], abstractMap)
    testData = []
    # Now go through a bunch of random places to put the monkey and ask the user which way to go
    for ii in range(N):
        blocked = True
        while blocked :
            position = (randrange(len(abstractMap)), randrange(len(abstractMap[0])))
            if abstractMap[position[0]][position[1]] == Roomgen.ABSTRACTIONS[' ']:
                blocked = False
        # Put the monkey there
        g.monkeys[0].setPos(position)
        # Get the surroundings
        surrVec, surrMap = g.surroundingVector(position, True)
        # Print it out
        print('Map')
        print(Roomgen.concretize(surrp, True))
        direction = input('>>>')
        if direction in list('wasd '): #Good input
            testData.append((direction, surrVec))
            file = open(filePath, 'a')
            file.write(str((direction,surrVec)))
            file.write('\n')
            file.close()
        else:
            print('Bad input')
    return testData

def generateTrainingDataContinuous(N,filePath,abstractMap,saturateFood=False):
    # This is a creator for generating data from the map
    # The monkey retains its location between ticks
    # Create a monkey
    testMonkey = Monkey.Monkey()
    # Create the grid
    g = Grid.Grid([testMonkey], [(3,3)], abstractMap)
    testData = []
    # Find a place to put the monkey
    blocked = True
    while blocked :
        position = (randrange(len(abstractMap)), randrange(len(abstractMap[0])))
        if abstractMap[position[0]][position[1]] == Roomgen.ABSTRACTIONS[' ']:
            blocked = False
    # Put the monkey there
    g.monkeys[0].setPos(position)
    print('Monkey initialized at', position)
    if saturateFood: g.monkeys[0].food = (N+10)*foodPerTurn
    # Go for N turns
    for ii in range(N):
        # Tick
        try:
            direction, surrVec = g.tick(trainingData = True)
        except Exceptions.DeathError:
            print('Exiting training, returning test data')
            return testData
        # Write data
        directionVector = [0]*len(Roomgen.WASD)
        directionVector[Roomgen.WASD.index(direction)] = 1
        testData.append((directionVector, float(g.monkeys[0].food), surrVec))
        file = open(filePath, 'a')
        file.write(repr(testData[-1]).replace('\n','').replace('tensor','torch.tensor').replace(' ',''))
        file.write('\n')
        file.close()
    return testData

def train(brain, filePath, N, lr = 1e-2):
    """
    Trains a brain.

    Args:
        brain: The brain to be trained.
        filePath: The path to the file with the training data
        N: Number of passes through the training dataset.
        lr: Learning rate
    Returns:
        1: A list of all the loss functions (size N)
    Raises:
        FileNotFoundError: If filePath does not point to a valid file.
        SyntaxError: If the file is not formatted correctly.
        DataShapeError: If the training data does not properly fit the brain.
    """
    # Open the data file.
    inF = open(filePath, 'r')

    # Construct a loss function.
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