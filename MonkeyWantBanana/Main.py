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
					'#                      #         #\n'+\
					'#             b        #     b   #\n'+\
					'#    b                  d        #\n'+\
					'#                      #         #\n'+\
					'###########     b      #    d    #\n'+\
					'#          #           #         #\n'+\
					'#           #      ########    ###\n'+\
					'#    b           #      #        #\n'+\
					'#               d                #\n'+\
					'#              #                 #\n'+\
					'#      d      #    dd      b     #\n'+\
					'#    b d     #                   #\n'+\
					'#      d             b   d     ###\n'+\
					'#               #           ######\n'+\
					'##################################'

roomStart = Roomgen.abstract(roomStartPicture)



# Instantiate brain
brain = Brain.BrainDQN()

##########

s = torch.tensor([19.0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0])
brain.pi(s)

##########


# # Load brain from permanent memory
# brain.load_state_dict(torch.load('brainsave.txt'))


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
