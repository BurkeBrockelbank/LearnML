# This is the main program for Monkey Want Banana
from __future__ import print_function
from __future__ import division

import Roomgen
import Monkey
import Grid
import Trainer
import Brain
import torch
import matplotlib.pyplot as plt

# Start with the basic room
roomStartPicture = 	'##################################\n'+\
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

roomStart = Roomgen.abstract(roomStartPicture)

# Define number of turns of memory
memoryLength = 5
# Instantiate brain
brain = Brain.BrainDQN(memoryLength)
# Instantiate the monkey
mitch = Monkey.MonkeyDQN(brain)
g = Grid.Grid([mitch],[(13,18)],roomStart)

# # Load brain from permanent memory
# brain.load_state_dict(torch.load('brainsave.txt'))

# Train the monkey
Trainer.trainDQN(500000, g, 0.9, 0.8, 0.2, 5000, memoryLength, lr=0.01,loud=False,showEvery=1000)

# Save the brain to permanent memory
torch.save(brain.state_dict(), 'brainsave.txt')


# # Train the monkey
# for ii in range(1):
# 	learningRate = 1e-2
# 	epochs = 10
# 	prediction, loss, reportRecord = Trainer.train(brain, 'Data1111.txt', \
# 						epochs, lr=learningRate, reports=3, quiet=False)

# 	# Save the brain to permanent memory
# 	torch.save(brain.state_dict(), 'brainsave.txt')

# 	# Save the report record
# 	outF = open('report.txt', 'a')
# 	outF.write(str(reportRecord))
# 	outF.write('\n')
# 	outF.close()
# 	# Load the old reports
# 	toShow = Trainer.loadRecords('report.txt')
# 	# Plot the report record
# 	plt.plot(*zip(*toShow))
# 	plt.savefig('./img/brain2.png')
# 	plt.show()
# 	plt.clf()


# # Generate training data
# mitch = Monkey.Monkey(brain)
# g = Grid.Grid([mitch],[(3,10)],roomStart)
# Trainer.generateTrainingDataContinuous(1000, 'Data1111.txt', roomStart)

# # Test the monkey
# mitch = Monkey.Monkey(brain)
# g = Grid.Grid([mitch],[(10,16)],roomStart)
# for i in range(50):
# 	g.tick(wait=True)
