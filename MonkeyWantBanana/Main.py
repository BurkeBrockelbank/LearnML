# This is the main program for Monkey Want Banana
from __future__ import print_function
from __future__ import division

import Roomgen
import Monkey
import Grid

# Start with the basic room
roomStartPicture = 	'##################################\n'+\
					'#                                #\n'+\
					'#     ##                         #\n'+\
					'#    b                           #\n'+\
					'#                                #\n'+\
					'#                                #\n'+\
					'#                                #\n'+\
					'#                                #\n'+\
					'#                                #\n'+\
					'#                                #\n'+\
					'#                                #\n'+\
					'#                          b     #\n'+\
					'#                                #\n'+\
					'#                                #\n'+\
					'#                                #\n'+\
					'##################################'

roomStart = Roomgen.abstract(roomStartPicture)



Mitch = Monkey.Monkey()
Mitch.food = 10000
MitchGrid = Grid.Grid([Mitch],[(2,3)],roomStart)
print(MitchGrid.invisibleCone(5,(5,5),(1,5)))
for i in range(50): MitchGrid.tick()