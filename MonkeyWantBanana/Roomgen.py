# This is the room generator program. For now, we are just using a empty square room.
# I want to be able to draw a room with ascii art and have it become an abstraction,
# where it will be manipulated and then converted back into ascii art to be read by
# the user.

from __future__ import print_function
from __future__ import division

# Define the ABSTRACTIONS
ABSTRACTIONS = {'?': -1,#unknown
					'#':0,#Barrier
					'm':1,#Monkey
					' ':2,#Empty space
					'b':3,#Banana
					}
CONCRETIZATIONS = dict()
for key in ABSTRACTIONS.keys():
	CONCRETIZATIONS[ABSTRACTIONS[key]] = key

# Make the abstract and concretize functions
def abstract(mapPicture):
	rows = mapPicture.split('\n')
	abstractRows = []
	for row in rows:
		abstractRow = []
		for char in row:
			abstractRow.append(ABSTRACTIONS[char])
		abstractRows.append(list(abstractRow))
	return abstractRows

def concretize(mapList):
	mapPicture = ''
	for abstractRow in mapList:
		row = ''
		for integer in abstractRow:
			row += CONCRETIZATIONS[integer]
		mapPicture += row+'\n'
	return mapPicture[:-1]

