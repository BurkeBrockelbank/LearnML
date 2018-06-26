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
     # Import the ASCII map.
     room_start = rg.ASCII_to_channel(gl.room_start_ASCII)
     # Build grid object with monkey and brain
     monkey_brain = brain.BrainDQN()
     monkey = monkey.Monkey(monkey_brain)
     monkey.pos = (3,3)
     g = grid.Grid([monkey], room_start)

     # # Generate training data
     # train.training_data(1000,['throwaway.txt'], g)

     # Supervised monkey training
     dump_parameters(monkey_brain, 'brain0.txt')
     train.supervised_training(50, ['data_channels.txt'], monkey_brain, 0.8, 0.01)
     dump_parameters(monkey_brain, 'brain1.txt')

     # # Test the user control of the monkey.

     # # Define number of turns of memory
     # memoryLength = 15
     # # Define discount factor
     # gamma = 0.8
     # # Instantiate brain
     # brain = Brain.BrainDQN(memoryLength)
     # # Instantiate the monkey
     # mitch = Monkey.MonkeyDQN(brain)
     # g = Grid.Grid([mitch],[(13,18)],roomStart)


     # # Load brain from permanent memory
     # brain.load_state_dict(torch.load('brainsave.txt'))

     # # Train the monkey
     # for ii in range(10):
     #     print(ii)
     #     learningRate = 1e-3
     #     epochs = 1000
     #     reports = 20
     #     loss, reportRecord = Trainer.trainDQNSupervised(brain, 'Data1111.txt', epochs, memoryLength, gamma, \
     #     lr = 1e-3, reports = reports, quiet = False)

     #     # Save the brain to permanent memory
     #     torch.save(brain.state_dict(), 'brainsave.txt')

     #     # Save the report record
     #     outF = open('reportDQN.txt', 'a')
     #     outF.write(str(reportRecord))
     #     outF.write('\n')
     #     outF.close()
     #     # Load the old reports
     #     toShow = Trainer.loadRecords('reportDQN.txt')
     #     # Grab the most recent epoch and separate it (to colour it differently)
     #     olderData = toShow[:-reports]
     #     newerData = toShow[-reports:]
     #     # Plot the report record
     #     plt.title('Supervised Portion of DQN Transfer Learning')
     #     plt.xlabel('Epochs')
     #     plt.ylabel('Quality Loss (Sum of Squares)')
     #     plt.plot(*zip(*olderData), color='blue')
     #     plt.plot(*zip(*newerData), color='red')
     #     plt.savefig('./img/brainDQNSupervised.png')
     #     plt.clf()


     # # Generate training data
     # Trainer.generateTrainingDataContinuous(2000, 'DataDQNTransfer.txt', roomStart)

     # # Test the monkey
     # mitch = Monkey.Monkey(brain)
     # g = Grid.Grid([mitch],[(10,16)],roomStart)
     # for i in range(50):
     #     g.tick(wait=True)
