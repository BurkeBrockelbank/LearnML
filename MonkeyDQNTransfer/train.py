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


def monkey_training_data(N, paths, g, loud=[]):
    """
    This generates training data based on the actions of a monkey. The
    intention is for this to be used with an A.I.

    Args:
        N: The number of ticks in the training data.
        paths: A list of paths leading to the data files. One path must be
            present for each monkey in the grid.
        g: The grid to generate training data from.
        loud: Default []. List of monkey indeces to watch.
    """
    charity = False
    for n in range(N):
        print('Turn', n, 'begun.')
        # Tick the monkeys
        foods, actions, surroundings = g.tick(0, invincible = True, \
            loud=loud)
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
                    # See if the monkey has starved to death before.
                    if charity:
                        # We should teleport the monkey to get it unstuck
                        charity = False
                        invalid_spot = True
                        while invalid_spot:
                            i = random.randrange(g.width)
                            j = random.randrange(g.height)
                            # This spot can have a monkey placed on it
                            if g.channel_map[gl.INDEX_BARRIER,i,j] == 0 and \
                                g.channel_map[gl.INDEX_DANGER,i,j] == 0:
                                # First teleport the monkey on the channel map
                                g.teleport_monkey(monkey.pos, (i,j))
                                # Update the position in the monkey object
                                monkey.pos = (i,j)
                                # Flag for completion.
                                invalid_spot = False
                    else:
                        charity = True


def training_data(N, paths, g,loud=[]):
    """
    This generates training data for the monkey with user input.
    
    Args:
        N: The number of ticks in the training data.
        paths: A list of paths leading to the data files. One path must be
            present for each monkey in the grid.
        g: The grid to generate training data from.
        loud: Default []. List of monkey indeces to watch.
    """
    for n in range(N):
        # Tick the monkeys
        foods, actions, surroundings = g.tick(2, loud=loud)
        # Iterate through the paths, surroundings, and actions
        for path, food, action, surr in zip(paths, foods, actions, surroundings):
            # Write the data to file
            outF = open(path, 'a')
            outF.write('(')
            outF.write(str(food))
            outF.write(',')
            outF.write(str(int(action)))
            outF.write(',')
            surr_string = str(surr)
            surr_string = surr_string.replace('tensor','torch.tensor')
            surr_string = surr_string.replace(' ','')
            surr_string = surr_string.replace('\n','')
            outF.write(surr_string)
            outF.write(')')
            outF.write('\n')
            outF.close()

def supervised_training(epochs, batches, paths, brain, gamma, \
    max_discount, lr, reports, intermediate = ''):
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
        reports: The number of times to print progress.
        intermediate: The file to save intermediate brain trainings to.


    Returns:
        0: Training data in the form of list of tuples. First element is epoch
        number, second number is average loss over this epoch.
    """
    # Set the brain to training mode
    brain.train()

    all_data = []

    # First read all training data
    all_lines = []
    for path in paths:
        print('Reading', path)
        in_f = open(path, 'r')
        in_lines = in_f.readlines()
        in_f.close()
        # parse the input lines
        data = [eval(x.rstrip()) for x in in_lines]
        all_lines.append(data)
        # As a reminder, the data structure is
        # food (int), action (int),board state (torch.tensor dtype=torch.uint8)
        # Now we need to calculate the quality for each of these
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
        new_data = [(torch.tensor(quality),) + state_tuple \
            for state_tuple, quality in zip(new_data, quals)]
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

    # Calculate batch data
    batch_length = len(data_set)//batches

    # Due to symmetry, we can increase the data set eightfold.
    data_symmetric = []
    # Still unimplemented

    # Report status
    print('Data loaded')

    # Now we do the actual learning!
    # Define the loss function
    criterion = nn.SmoothL1Loss(size_average=False)
    # Create an optimizer
    optimizer = torch.optim.Adam(brain.parameters(), lr=lr)
    loss_record = []
    # Iterate through epochs
    for epoch in range(epochs):
        # Permute the data to decorrelate it.
        random.shuffle(data_set)
        # Separate into batches
        batched_data = []
        for batch_no in range(batches-1):
            batch_start = batch_no*batch_length
            batched_data.append(data_set[batch_start:batch_start+batch_length])
        batched_data.append(data_set[(batches-1)*batch_length:])
        # See if we are reporting this time
        if epoch%(epochs//reports) == epochs%(epochs//reports):
            report_this = True
            total_loss = 0
        else:
            report_this = False

        # Iterate through data
        for batch_no, batch_set in enumerate(batched_data):
            if report_this:
                print('Epoch', epoch, 'Batch', batch_no, 'begun')
            for real_Q, food, action, vision in batch_set:
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

        # Save brain
        if intermediate != '':
            torch.save(brain.state_dict(), intermediate)

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


def curated_bananas_dqn(g, level, N, gamma, lr, food,\
    epsilon = lambda x: 0, watch = False):
    """
    This function trains the monkey with curated examples. For example,
    at level 1, we will have a banana one block away from the monkey:
     b       b       b                                         
      m      m      m     bm     mb     m      m      m             
                                       b       b       b 
    Level two has the banana 2 or 1 away and so on.

    Args:
        g: The grid with a single monkey to train. Channel map will be
            rewritten.
        level: The level of curation to train.
        N: The number of iterations of training to do.
        lr: learning rate.
        food: The maximum food level. All food levels will be used randomly
            up to this level.
        epsilon_data: Default lambda function returning zero. A function that
            calculates epsilon.
        watch: Default False. True if you want to watch the monkey train.
    """
    # Build the channel map
    map_size = len(gl.SIGHT)+4*level
    radius = map_size//2
    g.channel_map = torch.zeros((len(gl.BLOCK_TYPES), map_size, map_size), \
        dtype = torch.uint8)
    # Build walls
    for i in range(map_size):
        for j in [0, -1]:
            # Place barrier
            g.channel_map[gl.INDEX_BARRIER, i, j] = 1
            # Place barrier in transpose position
            g.channel_map[gl.INDEX_BARRIER, j, i] = 1


    # Place monkey for new channel map
    g.monkeys[0].pos = (0,0)
    g.replace_monkeys()

    # Update width and height of channel map
    g.width = len(g.channel_map)
    g.height = len(g.channel_map[0])

    # Unpack epsilon if it exists
    epsilon_needed = False
    if g.monkeys[0].brain.pi == g.monkeys[0].brain.pi_epsilon_greedy:
        epsilon_needed = True

    # Initialize banana position variables
    banana_i = 0
    banana_j = 0

    # Initialize loss record
    loss_record = []

    # Calculate the number of turns the monkey has to get banana.
    turn_allowance = math.ceil((2*level**2 + 3*level + 1)/(2*level + 2))

    for n in range(N):
        if n%100 == 0:
            print('Curated training', n, 'epsilon', epsilon(n))
        # Assign food
        g.monkeys[0].food = random.randrange(2*level,food)

        # Remove old bananas
        g.channel_map[gl.INDEX_BANANA] = torch.zeros((map_size, map_size), \
            dtype = torch.uint8)

        # Assign banana position
        position_chosen = False
        while not position_chosen:
            banana_i = random.randrange(radius-level, radius+level+1)
            banana_j = random.randrange(radius-level, radius+level+1)
            if (banana_i, banana_j) != (radius, radius):
                position_chosen = True
        g.channel_map[gl.INDEX_BANANA, banana_i, banana_j] = 1

        # Replace monkey
        g.teleport_monkey(g.monkeys[0].pos, (radius,radius))
        g.monkeys[0].pos = (radius,radius)

        # Calculate epsilon
        if epsilon_needed:
            epsilon_pass = lambda x : epsilon(n)
        else:
            epsilon_pass = epsilon

        # Do a run of dqn training on monkey
        loss_record_1 = dqn_training(g, turn_allowance, gamma, lr, \
            epsilon = epsilon_pass, watch = watch)
        loss = sum(list(zip(*loss_record_1))[1])
        loss_record.append((n, loss))

        # Reset turn counter
        g.turn_count = 0

    return loss_record




def dqn_training(g, N, gamma, lr, \
    epsilon = lambda x: 0, watch = False):
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
        epsilon: Default lambda function returning zero. A function that gives
            the value for epsilon based on epoch number.
        lr: The learning rate.
        watch: Default False. If True, will wait for the user to look at every
            iteration of the training.

    Returns:
        0: Training data in the form of list of tuples. First element is
        iteration number, second number is average loss over the
        iterations leading up to this report.
    """
    # Determine if we want to watch
    if watch:
        loud = [0]
    else:
        loud = []

    # Unpack epsilon if it exists
    epsilon_needed = False
    if g.monkeys[0].brain.pi == g.monkeys[0].brain.pi_epsilon_greedy:
        epsilon_needed = True

    # Instantiate total reward
    total_reward = 0

    # Calculate the state for the first time.
    g.monkeys[0].brain.eval()
    sight_new = g.surroundings(g.monkeys[0].pos)
    food_new = g.monkeys[0].food
    state_new = (food_new, sight_new)
    if epsilon_needed:
        Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new, epsilon(0))
    else:
        Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new)
    g.monkeys[0].brain.train()


    # Define optimizer
    optimizer = torch.optim.Adam(g.monkeys[0].brain.parameters(), lr=lr)
    # Define loss criterion
    criterion = nn.SmoothL1Loss(size_average=False)


    loss_record = []

    # Iterate N times
    for n in range(N):
        if watch:
            print('-----------------------')

        # 1) Get the policy's action.
        Q = Q_new
        a = a_new
        p = p_new

        # 2) Get the consequent state (move the monkey).
        g.tick(1, directions = [a], invincible = True, loud=loud, wait=False)
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
            Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new, epsilon(n))
        else:
            Q_new, a_new, p_new = g.monkeys[0].brain.pi(state_new)
        # c) Calculate the loss difference
        delta = Q - r - gamma * Q_new
        # d) Calculate the loss as Huber loss.
        loss = criterion(delta, torch.zeros(1))
        loss_record.append((n,float(loss)))

        if watch:
            print(gl.WASD[a], 'with probability', p)
            print('Q(s,', gl.WASD[a], ') = ', round(float(Q),3), sep='')
            print('--> r + gamma * Q(s\',', gl.WASD[a_new], ')', sep='')
            print('  = ', r, ' + ', gamma, ' * ', round(float(Q_new),3), sep='')
            print('  = ', round(float(r+gamma*Q_new),3), sep='')
            print('delta = ' + str(float(delta)))
            input('loss = ' + str(float(loss)))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward(retain_graph= (n!=N-1))
        optimizer.step()

    return loss_record

def test_model(g, N, reset):
    """
    This function will test the first monkey in the grid given for its
    score in the game after N resets of some number of turns each. The
    score is the sum of the food at the end of each reset.

    Args:
        g: The grid that the monkeys are on.
        N: The number of resets to do.
        reset: The number of turns per reset.

    Returns:
        0: Average score over all resets.
    """
    # Set all monkeys to evaluation mode
    for monkey in g.monkeys:
        monkey.brain.eval()

    # Initialize score.
    total_score = 0

    # Iterate over resets.
    for n in range(N):
        print('Reset', n)

        # Randomize position
        invalid_spot = True
        while invalid_spot:
            i = random.randrange(g.width)
            j = random.randrange(g.height)
            # This spot can have a monkey placed on it
            if g.channel_map[gl.INDEX_BARRIER,i,j] == 0 and \
                g.channel_map[gl.INDEX_DANGER,i,j] == 0:
                # First teleport the monkey on the channel map
                g.teleport_monkey(g.monkeys[0].pos, (i,j))
                # Update the position in the monkey object
                g.monkeys[0].pos = (i,j)
                # Flag for completion.
                invalid_spot = False

        # Set food value
        g.monkeys[0].food = reset+1

        # Revive monkey if dead
        g.monkeys[0].dead = False

        # Do turns
        for turn in range(reset):
            g.tick(0, invincible = True)
            # Check if dead
            if g.monkeys[0].dead:
                # No score contribution
                g.monkeys[0].food = 0
                break

        # Reset turn number
        g.turn_count = 0
        g.monkeys[0].age = 0

        # Add to score
        total_score += g.monkeys[0].food

    return total_score/N