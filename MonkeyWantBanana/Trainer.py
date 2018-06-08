# This impliments some supervised training for the monkey.
#

from __future__ import print_function
from __future__ import division
import Roomgen
import Monkey
import Grid
import Brain
import Exceptions
from random import randrange
import torch

def generateTrainingDataSporadic(N, filePath,abstractMap):
    # This is a creator for generating data from the map
    # The monkey starts at a random location each tick
    # Create a monkey
    b = Brain.Brain0()
    testMonkey = Monkey.Monkey(b)
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
        if direction in list(Roomgen.WASD): #Good input
            directionVector = [0]*len(Roomgen.WASD)
            directionVector[Roomgen.WASD.index(direction)] = 1
            testData.append((directionVector, float(g.monkeys[0].food), surrVec.tolist()))
            file = open(filePath, 'a')
            file.write(repr(testData[-1]).replace('\n','').replace(' ',''))
            if ii != N-1:
                file.write('\n')
            file.close()
        else:
            print('Bad input')
    return testData

def generateTrainingDataContinuous(N,filePath,abstractMap,saturateFood=False):
    """
    This is a creator for generating data from the map. The monkey retains its
    location between ticks. The format of the data file is a one-hot integer
    5-list, followed by a float, followed by a list of length corresponding
    to the number of visible spaces in SIGHT.

    Args:
        N: Number of turns allowed in simulation.
        filePath: The path to the output data file.
        abstractMap: An abstracted map (passed through Roomgen.abstract).
        saturateFood: Default False. If True, enough food will be given to the
        to the monkey so it won't die of hunger during the test.

    Returns:
        0: List of all the lines in the output file with not newline character
        at the end                 
    """
    # Create a monkey
    b = Brain.Brain0()
    testMonkey = Monkey.Monkey(b)
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
    if saturateFood: g.monkeys[0].food = (N+10)*g.monkeys[0].foodPerTurn
    # Go for N turns
    for ii in range(N):
        # Tick
        try:
            direction, surrVec = g.tick(trainingData = True, manualControl = True)
        except Exceptions.DeathError:
            print('Exiting training, returning test data')
            return testData
        # Write data
        directionVector = [0]*len(Roomgen.WASD)
        directionVector[Roomgen.WASD.index(direction)] = 1.0
        testData.append((directionVector, float(g.monkeys[0].food), surrVec.tolist()))
        file = open(filePath, 'a')
        file.write(repr(testData[-1]).replace('\n','').replace(' ',''))
        if ii != N-1:
            file.write('\n')
        file.close()
    return testData

def train(brain, filePath, N, lr = 1e-2, reports = 10, quiet = True):
    """
    Trains a brain.

    Args:
        brain: The brain to be trained.
        filePath: The path to the file with the training data
        N: Number of epochs.
        lr: Learning rate.
        reports: The number of times to report the progress

    Returns:
        1: A list of all the loss functions (size N)

    Raises:
        FileNotFoundError: If filePath does not point to a valid file.
        SyntaxError: If the file is not formatted correctly.
        DataShapeError: If the training data does not properly fit the brain.
    """

    # Determine the remainder at which we want to print the iteration number
    if reports != 0:
        reportEvery = N//reports
    # Create a list for the reports
    reportList = []
    # Open the data file and read the data
    inF = open(filePath, 'r')
    dataLines = []
    for line in inF:
        dataLines.append(eval(line.rstrip()))
    inF.close()
    # Put the data into input (x) and output (y) 2d tensors. Each row is one
    # data point (sight+food+movement)
    x = []
    y = []
    for tup in dataLines:
        y.append(tup[0])
        x.append([tup[1]]+tup[2])
    x = torch.tensor(x)
    # y is a list of 1-hot lists. Let's switch this over to labels
    y1 = torch.tensor([foo.index(1) for foo in y])
    # Also get y
    y = torch.FloatTensor(y)

    # Define the loss function
    criterion = torch.nn.CrossEntropyLoss()
    # Create an optimizer
    optimizer = torch.optim.SGD(brain.parameters(), lr)
    for epoch in range(N):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = brain(x)
        # Compute and print loss
        loss = criterion(y_pred, y1)
        if reports != 0:
            if (N-epoch-1)%(reportEvery) == 0:
                if not quiet:
                    print(epoch, loss.item())
                reportList.append((epoch, loss.item()))
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return y_pred, loss, reportList

def loadRecords(path):
    """
    Loads in the records for loss function vs. epochs
    Args:
        path: The path to the record file.
    Returns:
        0: A list of tuples of the form (epochs, loss)
    """
    records = []
    inF = open(path, 'r')
    for line in inF:
        records.append(eval(line.rstrip()))
    inF.close()
    
    # Update the epoch numbers in the records
    for i in range(1, len(records)):
        startEpoch = records[i-1][-1][0]
        newEpochs = []
        for point in records[i]:
            newEpochs.append((point[0]+startEpoch, point[1]))
        records[i] = newEpochs
    # Join the records together
    if len(records) == 1:
        return records
    else:
        return sum(records)