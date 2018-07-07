"""
This is the trainer module. It includes functions for generating training data and
performing supervised training on the brain.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/trainer.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import global_variables as gl
import exceptions
import room_generator as rg

import math
import random


def monkey_training_data(N, paths, g):
    """
    This generates training data based on the actions of a monkey. The
    intention is for this to be used with an A.I.

    Args:
        N: The number of ticks in the training data.
        paths: A list of paths leading to the data files. One path must be
            present for each monkey in the grid.
        g: The grid to generate training data from.
    """
    for n in range(N):
        print('Turn', n, 'begun.')
        # Tick the monkeys
        foods, actions, surroundings = g.tick(0, invincible = True, \
            loud=False)#, wait=False)
        # Iterate through the paths, surroundings, and actions
        for path, food, action, surr in zip(paths, foods, actions, surroundings):
            # Write the data to file
            outF = open(path, 'a')
            outF.write('(')
            outF.write(str(food))
            outF.write(',')
            outF.write(str(action))
            outF.write(',')
            surr_string = str(surr)
            surr_string = surr_string.replace('tensor','torch.tensor')
            surr_string = surr_string.replace(' ','')
            surr_string = surr_string.replace('\n','')
            outF.write(surr_string)
            outF.write(')')
            outF.write('\n')
            outF.close()

        # If the monkey died:
        for monkey in g.monkeys:
            if monkey.dead:
                monkey.dead = False
                # If the monkey needs food, give it a few bananas.
                if monkey.food < 0:
                    monkey.eat(5)

def training_data(N, paths, g):
    """
    This generates training data for the monkey with user input. Only tracks
    the 
    
    Args:
        N: The number of ticks in the training data.
        paths: A list of paths leading to the data files. One path must be
            present for each monkey in the grid.
        g: The grid to generate training data from.
    """
    for n in range(N):
        # Tick the monkeys
        foods, actions, surroundings = g.tick(2, loud=True)
        # Iterate through the paths, surroundings, and actions
        for path, food, action, surr in zip(paths, foods, actions, surroundings):
            # Write the data to file
            outF = open(path, 'a')
            outF.write('(')
            outF.write(str(food))
            outF.write(',')
            outF.write(str(action))
            outF.write(',')
            surr_string = str(surr)
            surr_string = surr_string.replace('tensor','torch.tensor')
            surr_string = surr_string.replace(' ','')
            surr_string = surr_string.replace('\n','')
            outF.write(surr_string)
            outF.write(')')
            outF.write('\n')
            outF.close()

def supervised_training(epochs, paths, brain, gamma, max_discount, lr,reports):
    """
    This performs supervised training on the monkey. 
    
    Args:
        N: The number of epochs to run in training.
        paths: A list of paths leading to the data files.
        brain: The brain to train.
        gamma: The discount factor in the Bellman equation.
        max_discount: The maximum factor to allow for discount in calculating
        qualities.
        lr: The learning rate to use.

    Returns:
        0: Training data in the form of list of tuples. First element is epoch
        number, second number is average loss over this epoch.
    """
    # Set the brain to training mode
    brain.train()
    # First concatenate all the training data and decorrelate it by shuffling.
    all_lines = []
    for path in paths:
        in_f = open(path, 'r')
        in_lines = in_f.readlines()
        in_f.close()
        # parse the input lines
        data = [eval(x.rstrip()) for x in in_lines]
        all_lines.append(data)
    # As a reminder, the data structure is
    # food (int), action (int), board state (torch.tensor torch.uint8)
    # Now we need to calculate the quality for each of these
    all_data = []
    for data in all_lines:
        food_vals = [x[0] for x in data]
        # We now will subtract subsequent food values to get the change in food
        food_diffs = [food_vals[i]-food_vals[i-1] for i in \
            range(1,len(food_vals))]
        # Delete the final row of data because it has no food difference
        # that can be calculated 
        new_data = data[:-1]
        # Calculate qualities
        quals = [0]
        for food_diff in food_diffs[::-1]:
            quals.append(quals[-1]*gamma+food_diff)
        quals = quals[1:]
        quals = quals[::-1]
        # Insert the quality into the data
        new_data = [(torch.tensor(quality),) + state_tuple for state_tuple, quality \
            in zip(new_data, quals)]
        # Add to the list of data sets
        all_data.append(new_data)
    # Since the final quality values concatenate the series short, we should
    # cut those data points. We will arbitrarily decide to ignore rewards which
    # have a reduction in magnitute by the factor max_discount.
    n_to_cut = math.ceil(math.log(max_discount)/math.log(gamma))
    all_data = [x[:-n_to_cut] for x in all_data]
    # And now we have processed the data

    # Concatenate the data sets.
    data_set = [el for one_path in all_data for el in one_path]

    # Permute the data to decorrelate it.
    random.shuffle(data_set)

    # Now we do the actual learning!
    # Define the loss function
    criterion = nn.SmoothL1Loss(size_average=False)
    # Create an optimizer
    optimizer = torch.optim.RMSprop(brain.parameters(), lr=lr)
    loss_record = []
    # Iterate through epochs
    for epoch in range(epochs):
        # See if we are reporting this time
        if epoch%(epochs//reports) == epochs%(epochs//reports):
            report_this = True
            total_loss = 0
        else:
            report_this = False
        # Iterate through data
        for real_Q, food, action, vision in data_set:
            s = (food, vision)
            # Get the quality of the action the monkey did
            predicted_Q = brain.Q(s,action)
            # Calculate the loss
            loss = criterion(predicted_Q, real_Q)
            if loss > 1000:
                # There is some issue with the network occasionally spitting
                # out huge values. We will cap the maximum value. This is
                # done by recalculating the loss with something designed to
                # just be just 1000 away from the prediction. To get this
                # value, we need to pull the value from predicted_Q and
                # remove its needs_gradient property. This is done by casting
                # to a floating point number.
                raise RuntimeWarning('Loss has been calculated as ridiculous.')
                loss = criterion(predicted_Q, \
                    torch.FloatTensor(float(predicted_Q)-1000))
            # Zero the gradients
            optimizer.zero_grad()
            # perform a backward pass
            loss.backward()
            # Update the weights
            optimizer.step()
            # Add to total loss
            if report_this:
                total_loss += float(loss)
        # Add to loss record
        if report_this:
            loss_record.append((epoch, total_loss/len(data_set)))
            print('Epoch', epoch, 'loss', total_loss/len(data_set))

    return loss_record

def load_records(path):
    """
    Loads in the records for loss function vs. epochs
    Args:
        path: The path to the record file.
    Returns:
        0: A list of tuples of the form (epochs, loss)
    """
    records = []
    in_file = open(path, 'r')
    for line in in_file:
        records.append(eval(line.rstrip()))
    in_file.close()
    
    # Update the epoch numbers in the records
    for i in range(1, len(records)):
        start_epoch = records[i-1][-1][0]+1
        new_epochs = []
        for point in records[i]:
            new_epochs.append((point[0]+start_epoch, point[1]))
        records[i] = new_epochs
    # Join the records together
    if len(records) == 1:
        return records[0]
    else:
        return sum(records, [])


def dqn_training(g, N, gamma, lr, \
    epsilon_data = (0,0,0), watch = False):
    """
    This function trains a monkey with reinforcement learning.

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
        g: The grid containing a single monkey containing a brain of
            superclass Brain_DQN.
        N: The number of iterations of training to do.
        gamma: The discount for the Bellman equation.
        epsilon_data: A tuple giving the initial and final values for
            epsilon in an epsilon greedy policy as well as the decay rate.
        lr: The learning rate.
        watch: Default False. If True, will wait for the user to look at every
            iteration of the training.

    Returns:
        0: Training data in the form of list of tuples. First element is
        iteration number, second number is average loss over the
        iterations leading up to this report.
    """
    # Unpack epsilon if it exists
    epsilon_needed = False
    if epsilon_data != (0,0,0):
        epsilon_start, epsilon_end, n_epsilon = epsilon_data
        epsilon_needed = True

    # Instantiate total reward
    total_reward = 0

    # Calculate the state for the first time.
    g.monkeys[0].brain.eval()
    sight_new = g.surroundings(g.monkeys[0].pos)
    food_new = g.monkeys[0].food
    state_new = (food_new, sight_new)
    if epsilon_needed:
        Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new, epsilon_start)
    else:
        Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new)
    g.monkeys[0].brain.train()


    # Define optimizer
    optimizer = torch.optim.RMSprop(g.monkeys[0].brain.parameters(), lr=lr)


    # Iterate N times
    for n in range(N):
        if watch:
            print('-----------------------')

        # 1) Get the policy's action.
        Q = Q_new
        a = a_new
        p = p_new

        # 2) Get the consequent state (move the monkey).
        g.tick(1, directions = [a], invincible = True, loud=watch, wait=False)
        state_old = state_new
        sight_new = g.surroundings(g.monkeys[0].pos)
        food_new = g.monkeys[0].food
        state_new = (food_new, sight_new)

        # 3) Get the immediate reward.
        # Immediate reward is normally food difference.
        r = state_new[0]-state_old[0]
        # If the monkey is dead, it instead gets a large penalty
        if g.monkeys[0].dead:
            r = -50
            # If the monkey died of hunger, feed it.
            if g.monkeys[0].food < 0:
                g.monkeys[0].eat(5)
                state_new = (g.monkeys[0].food, sight_new)
            g.monkeys[0].dead = False
        total_reward += r

        # 4) Calculate the loss
        # a) Calculate the quality of the move undertaken
        # This was already done in part 1.
        # b) Calculate the maximum quality of the subsequent move
        if epsilon_needed:
            epsilon = (epsilon_start - epsilon_end)*math.exp(-(n+1)/n_epsilon)\
                + epsilon_end
            Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new, epsilon)
        else:
            Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new)
        # c) Calculate the loss difference
        delta = Q - r - gamma * Q_new
        # d) Calculate the loss as Huber loss.
        loss = torch.nn.functional.smooth_l1_loss(delta, torch.zeros(1))

        if watch:
            print(gl.WASD[a], 'with probability', p)
            print('had quality', Q, 'r', r)
            input('delta ' + str(delta))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward(retain_graph= (n!=N-1))
        optimizer.step()

    return total_reward



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
        if loud:
            print('------------------------------')
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