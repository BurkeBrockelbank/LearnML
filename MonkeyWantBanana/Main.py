# This is the main program for Monkey Want Banana
from __future__ import print_function
from __future__ import division

import Roomgen
import Monkey
import Grid
#import Trainer
import Brain

# Start with the basic room
roomStartPicture = 	'##################################\n'+\
					'#                      #         #\n'+\
					'#             b        #     b   #\n'+\
					'#    b                  b        #\n'+\
					'#                 b    #         #\n'+\
					'###########            #         #\n'+\
					'#          #           #         #\n'+\
					'#           #      ########    ###\n'+\
					'#    b           #      #        #\n'+\
					'#              b#                #\n'+\
					'#              #                 #\n'+\
					'#             #            b     #\n'+\
					'#    b       #                   #\n'+\
					'#                                #\n'+\
					'#                                #\n'+\
					'##################################'

roomStart = Roomgen.abstract(roomStartPicture)



# I have continuous data in ContinuousData.txt
brain = Brain.Brain0()
mitch = Monkey.Monkey(brain)
g = Grid.Grid([mitch],[(2,20)],roomStart)
for i in range(18):
    g.tick(wait=True)
