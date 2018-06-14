"""
Main program

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/main.py
"""

from __future__ import print_function
from __future__ import division

import torch as to
import torch.nn as nn
import matplotlib.pyplot as plt

import global_variables as gl
import exceptions
import room_generator as rg
import brain
import monkey


# Start with the basic room
roomStartPicture =  '##################################\n'+\
                    '#  b                   #  b      #\n'+\
                    '#          d  b        #     b   #\n'+\
                    '#    b   b              d        #\n'+\
                    '#                 b    #         #\n'+\
                    '###########     b      #    d    #\n'+\
                    '#          #           #         #\n'+\
                    '#  b        #      ########    ###\n'+\
                    '#    b           #      #        #\n'+\
                    '#          b    d     b        b #\n'+\
                    '#              #         b       #\n'+\
                    '#      d      #    dd      b     #\n'+\
                    '#    b d     #                   #\n'+\
                    '#      d             b   b     ###\n'+\
                    '#         b     #           ######\n'+\
                    '##################################'

roomStart = rg.ASCII_to_channel(roomStartPicture)
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

# # Train the monkey with deep Q learning
# Trainer.trainDQN(500, g, gamma, 0.2, 0.05, 100, memoryLength, lr=0.001,loud=True,showEvery=1000)

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
