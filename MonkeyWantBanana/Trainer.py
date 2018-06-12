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
import math

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

def trainDQNSupervised(brain, filePath, N, memoryLength, gamma, lr = 1e-2, reports = 10, quiet = True):
    """
    Trains a brain set up for deep Q learning in a supervised way with the test data.

    Args:
        brain: The brain to be trained.
        filePath: The path to the file with the training data
        N: Number of epochs.
        memoryLength: The number of frames of memory to keep.
        gamma: The discount for calculating the quality of a move.
        lr: Default 0.01. Learning rate.
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
    a = []
    for tup in dataLines:
        a.append(tup[0])
        x.append([tup[1]]+tup[2])
    x = torch.tensor(x)
    a = torch.tensor(a)

    # We need to group the training data into sets with memory.
    sValues = []
    for i in range(memoryLength, x.size()[0]):
        thisMemory = x[i-memoryLength:i]
        sValues.append(torch.cat(tuple(thisMemory)).float())
    s = torch.stack(sValues)

    # Calculate the immediate reward of each move
    rValues = [x[i][0]-x[i-1][0] for i in range(memoryLength, x.size()[0])]
    # Calculate the best guess for the quality of the move
    QValues = [rValues[-1]]
    for r in rValues[-2::-1]:
        QValues.append(gamma*QValues[-1]+r)
    QValues = QValues[::-1]
    # Find out where to stop (we don't want to train uless we have a quality
    # value within about 10% of what is actually true)
    turnsToCut = math.ceil(-1.0/math.log10(gamma))
    QValues = QValues[:-turnsToCut]
    Q = torch.stack(QValues)

    # We can cut away the unused turns of y
    a = a[memoryLength-1:-turnsToCut-1]
    # Cut away the unused turns of the states
    s = s[:-turnsToCut]
    print(a.size())
    print(a[:6])
    print(s.size())
    print(s[:6])
    print(torch.stack(rValues).size())
    print(torch.stack(rValues)[:10])
    print(Q.size())
    print(Q[:10])
    return None

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

def trainDQN(N, g, gamma, epsilon0, epsilonF, nEpsilon, memoryLength,
    lr = 0.01, loud = False, showEvery=1):
    """
    This function trains a monkey with a brain of class BrainDQN
    on a grid.

    The DQN algorihm:
    1) Get the policy's action.
    2) Get the consequent state (move the monkey).
    3) Get the immediate reward from the grid.
    4) Calculate the loss
        a) Calculate the quality of the move undertaken Q(s,a).
        b) Calculate the max_a Q(s',a) where s' is the consequent
           state of performing a from the state s.
        c) delta = Q(s,a) - r - gamma*max_a Q(s', a)
           where r is the immediate loss measured from the system.
        d) Loss is the Huber loss (smooth L1 loss) of delta.


    Args:
        N: The number of iterations of training to do.
        g: The grid containing a monkey containing a brain of class
            BrainDQN.
        gamma: The discount for the Bellman equation.
        epsilon0: The initial value for epsilon for the epsilon-greedy
            policy.
        epsilonF: The final value for epsilon.
        nEpsilon: The decay rate for epsilon.
        lr: Default 0.01. The learning rate.
        loud: Default False. If true, prints the game screen, and the loss
        for each iteration.
        showEvery: Set this to a natural number to show the progress every
            showEvery iterations.
    """
    # Define the optimizer
    optimizer = torch.optim.RMSprop(g.monkeys[0].brain.parameters(), lr=lr)
    totalReward = 0

    # Calculate state for the first time
    surroundingsPrime, _ = g.surroundingVector(g.monkeys[0].pos)
    # Next we need to get the monkey's food level.
    foodPrime = torch.tensor([g.monkeys[0].food])
    # Compile these into a state vector.
    s1Prime = torch.cat((foodPrime, surroundingsPrime))
    s1Prime = s1Prime.float()

    # Define the memory
    memory = []
    # If the memory is empty, just set the monkey to think it has
    # been sitting here for a few turns.
    for i in range(memoryLength):
        memory.append(s1Prime)
    sPrime = torch.cat(tuple(memory))

    # Iterate N times
    for n in range(N):
        # 1) Get the policy's action.
        # We will first compute epsilon
        epsilon = epsilonF + (epsilon0-epsilonF)*math.exp(-n/nEpsilon)
        # Next, we need to first get the state. This
        # comes from the previous iteration
        surroundings = surroundingsPrime
        food = foodPrime
        s = sPrime
        # Compute the policy's best action.
        a = g.monkeys[0].brain.pi(s, epsilon, loud=loud)
        # 2) Get the consequent state sPrime
        # First we need to convert the action to its string form so
        # we can interface with the grid.
        actionString = Roomgen.WASD[int(a.max(0)[1])]
        # Move the monkey.
        g.tick(control='manual', directions=[actionString], wait=True,
            quiet = not loud, invincible=True)
        # Give the monkey a banana for free if it died of hunger
        charity = False
        if g.monkeys[0].food < 0:
            g.monkeys[0].eat(1)
            charity = True
        # Get the new state, starting with vision
        surroundingsPrime, _ = g.surroundingVector(g.monkeys[0].pos)
        # Next we need to get the monkey's food level.
        foodPrime = torch.tensor([g.monkeys[0].food])
        # Compile these into a state vector.
        s1Prime = torch.cat((foodPrime, surroundingsPrime))
        s1Prime = s1Prime.float()
        # Update the memory
        memory.append(s1Prime)
        del memory[0]
        # Get the total state with memory 
        sPrime = torch.cat(tuple(memory))

        # 3) Determine the immediate reward.
        # If the monkey dies, give some crazy low reward
        if g.monkeys[0].dead:
            r = torch.tensor(-10000.0)
            # Need to recussitate the monkey
            g.monkeys[0].dead = False
            # If the monkey died of hunger, we won't penalize it too
            # much right now.
            if charity:
                r = torch.tensor(-50.0)
        else:
            # Otherwise the immediate reward is just the change in food.
            r = foodPrime - food
            # Make sure this is a float
            r = r.float()
        # Add the immediate reward to the total reward counter
        totalReward += float(r.view(1)[0])

        # 4) Calculate the loss
        # a) Find out what quality is assigned to the move that was taken.
        # Note: The tensors need to be reshaped from size (0,n) to size (1,n)
        Qsa = g.monkeys[0].brain(s.view(1,len(s)), a.view(1,len(a)))
        # Recast Q(s, a) to a 1-tensor
        Qsa = Qsa.view(1)
        # b) Calculate maxa = argmax_a Q(s', a)
        maxa = g.monkeys[0].brain.maxa(sPrime)
        # Calculate max_a Q(s', a)
        maxaQsPrimea = g.monkeys[0].brain(sPrime.view(1,len(sPrime)), \
            maxa.view(1,len(maxa)))
        # Cast max_a Q(s', a) to a 1-tensor
        maxaQsPrimea = maxaQsPrimea.view(1)
        # c) Calculate  delta
        delta = Qsa - r - gamma*maxaQsPrimea
        if loud or n%showEvery == 0:
            print(n,'/',N,sep='')
            # Display the delta value
            print('delta = Q(s,'+actionString+') - r - gamma * max_a Q(s\', a) =', \
                Qsa.item(), '-', r.item(), '-', gamma, '*', maxaQsPrimea.item(), \
                '=', delta.item())
            # Display the average reward per turn to date.
            print(totalReward/(n+1), 'rpt')
            # print('s', s[:20])
            # print('a', a)
            # print('s\'', sPrime[:20])
            # print('max_a', a)
        # d) Calculate loss
        loss = torch.nn.functional.smooth_l1_loss(delta, torch.zeros(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in g.monkeys[0].brain.parameters():
            param.grad.data.clamp_(-1,1)
        optimizer.step()