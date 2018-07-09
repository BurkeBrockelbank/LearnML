"""
Main program

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/main.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

import global_variables as gl
import exceptions
import room_generator as rg
import brain
import monkey
import grid
import train

def dump_parameters(brain, path):
    out_F = open(path, 'w')
    out_F.write(str(list(brain.parameters())))
    out_F.close()

if __name__ == "__main__":
    # Some constants we will be using
    gamma = 0.6
    lr_supervised = 0.001
    lr_reinforcement = 0.00001
    epochs = 5
    batches = 10
    reports = 5
    N = 500
    epsilon_start = 0.4
    epsilon_end = 0.02
    n_epsilon = 1000
    max_discount = 0.05

    # Import the ASCII map.
    room_start = gl.RAND_ROOM #rg.ASCII_to_channel(gl.ROOM_START_ASCII)
    # Create brain to train
    monkey_brain = brain.BrainV2()
    # Put brain in monkey in grid
    monkeys = [monkey.Monkey(monkey_brain)]
    monkeys[0].pos = (len(room_start[1])//2,len(room_start[2])//2)
    g = grid.Grid(monkeys, room_start)

    # Make data paths for the monkeys
    paths = ['AIDATA\\AIData'+str(i)+'.txt' for i in range(50)]

    # # Train the monkey
    # train_data = train.supervised_training(50, batches, paths, monkey_brain, \
    #     gamma, max_discount, lr_supervised, 50, intermediate='brain_intermediate')

    # # Save the brain to permanent memory
    # torch.save(monkey_brain.state_dict(), 'brainsave_v2.txt')

    # plt.title('Supervised Learning on RAND_ROOM lr' + str(lr_supervised))
    # plt.xlabel('Turn')
    # plt.ylabel('Loss')
    # plt.plot(*zip(*train_data))
    # plt.show()

    # # Build monkeys
    # monkeys = [monkey.Monkey(brain.BrainLinearAI()) for i in \
    #     range(50)]
    # for monkey in monkeys:
    #     # Place the monkeys a bit away from the walls (10 blocks).
    #     i = random.randrange(10,gl.RAND_ROOM_WIDTH-10)
    #     j = random.randrange(10,gl.RAND_ROOM_WIDTH-10)
    #     monkey.pos = (i,j)
    # g = grid.Grid(monkeys, room_start)


    # # Generate training data from the A.I.
    # train.monkey_training_data(1000000, paths, g, loud=[])

    # # Generate training data
    # train.training_data(1000,['throwaway.txt'], g)

    # # Load brain from permanent memory
    # monkey_brain.load_state_dict(torch.load('brainsave_v2.txt'))

    # # Supervised monkey training
    # dump_parameters(monkey_brain, 'brain0.txt')
    # loss_data = train.supervised_training(epochs, ['data_channels.txt'], \
    #     monkey_brain, gamma, lr, reports)
    # dump_parameters(monkey_brain, 'brain1.txt')

    # # Reinforcement monkey training
    # for i in range(100):
    #     total_reward = train.dqn_training(g, N, gamma, lr, \
    #         epsilon_data = (epsilon_start, epsilon_end, n_epsilon), \
    #         watch = False)
    #     print(i,total_reward)

    # Watch monkey train
    total_reward = train.dqn_training(g, 20, gamma, lr_reinforcement, \
        epsilon_data = (epsilon_start, epsilon_end, n_epsilon), \
        watch = True)

    for i in range(300):

        total_reward = train.dqn_training(g, N, gamma, lr_reinforcement, \
            epsilon_data = (epsilon_start, epsilon_end, n_epsilon), \
            watch = False)

        # Save the training data
        out_file = open('reward_report.txt', 'a')
        out_file.write(str(loss_data))
        out_file.write('\n')
        out_file.close()
        # Load the old reports
        to_show = train.loadRecords('reward_report.txt')
        # Grab the most recent epoch and separate it (to colour it differently)
        older_data = to_show[:-reports]
        newer_data = to_show[-reports:]
        # Plot the report record
        plt.title('Supervised Portion of DQN Transfer Learning')
        plt.xlabel('Epochs')
        plt.ylabel('Quality Loss (Sum of Squares)')
        plt.plot(*zip(*older_data), color='blue')
        plt.plot(*zip(*newer_data), color='red')
        plt.savefig('./img/reinforcement.png')
        plt.clf()

        # Test monkey
        g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_greedy
        print(train.test_model(g, 100, 30))
        g.monkeys[0].brain.pi = g.monkeys[0].brain.pi_epsilon_greedy
