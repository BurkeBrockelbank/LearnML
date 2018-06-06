# This is the room generator program. For now, we are just using a empty square room.
# I want to be able to draw a room with ascii art and have it become an abstraction,
# where it will be manipulated and then converted back into ascii art to be read by
# the user.

from __future__ import print_function
from __future__ import division

# Define the ABSTRACTIONS
# ? Unknown
# # Barrier
# m Monkey
#   Empty space
# b Banana
# d Danger
BLOCKTYPES = ['?', '#', 'm', ' ', 'b', 'd']
ABSTRACTS = [-1, 0, 1, 2, 3, 4]
ABSTRACTIONS = dict()
CONCRETIZATIONS = dict()
for key, value in zip(BLOCKTYPES, ABSTRACTS):
    ABSTRACTIONS[key] = value
    CONCRETIZATIONS[value] = key
WASD = 'wasd '

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

def concretize(mapList,indeces=False,indexOffset=(0,0)):
    mapPicture = ''
    if indeces:
        d10 = lambda x : x-x%10
        mapPicture = ' '*len(str(d10(indexOffset[0])))+'  '+str(d10(indexOffset[1]))+'+\n '+\
                    ' '*len(str(d10(indexOffset[0])))+' '
        for i in range(len(mapList[0])):
            mapPicture += str((indexOffset[1]+i)%10)
        mapPicture += '\n'
    for i, abstractRow in enumerate(mapList):
        row = ''
        for integer in abstractRow:
            row += CONCRETIZATIONS[integer]
        if indeces:
            if i==0:
                mapPicture += str(d10(indexOffset[0]))+'+'
            else:
                mapPicture += ' '*len(str(d10(indexOffset[0])))+' '
            mapPicture += str((indexOffset[0]+i)%10)
        mapPicture += row+'\n'
    mapPicture = mapPicture[:-1] #Strip the last newline character
    return mapPicture

