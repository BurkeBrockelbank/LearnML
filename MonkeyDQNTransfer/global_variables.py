"""
This file contains all global variables used for running the program.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/global_variables.py
"""

from __future__ import print_function
from __future__ import division

import torch
import random

# SIGHT =[[0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
#         [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
#         [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
#         [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
#         [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0]]

# SIGHT =[[0,0,0,1,0,0,0],
#         [0,1,1,1,1,1,0],
#         [0,1,1,1,1,1,0],
#         [1,1,1,1,1,1,1],
#         [0,1,1,1,1,1,0],
#         [0,1,1,1,1,1,0],
#         [0,0,0,1,0,0,0]]

# SIGHT =[[0,0,0,1,1,1,1,1,0,0,0],
#         [0,0,1,1,1,1,1,1,1,0,0],
#         [0,1,1,1,1,1,1,1,1,1,0],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [0,1,1,1,1,1,1,1,1,1,0],
#         [0,0,1,1,1,1,1,1,1,0,0],
#         [0,0,0,1,1,1,1,1,0,0,0]]
SIGHT11 = torch.ones((11,11), dtype=torch.uint8)


SIGHT5 = torch.ones((5,5), dtype=torch.uint8)
# Note: SIGHT must be square and have an uneven number of rows.
SIGHT = SIGHT11

# Define the block types
BLOCK_TYPES = ['#', 'm', 'b', 'd']
EMPTY_SYMBOL = ' '

# Get indeces for the block types
INDEX_BARRIER = BLOCK_TYPES.index('#')
INDEX_MONKEY = BLOCK_TYPES.index('m')
INDEX_BANANA = BLOCK_TYPES.index('b')
INDEX_DANGER = BLOCK_TYPES.index('d')

# Define movement symbols
WASD = 'wasd '

# Start with the basic room
ROOM_START_ASCII =  '##################################\n'+\
                    '#  b                   #  b      #\n'+\
                    '#          d  b        #     b   #\n'+\
                    '#    b                  d        #\n'+\
                    '#                 b    #         #\n'+\
                    '###########     b      #    d  b #\n'+\
                    '#          #           #  b      #\n'+\
                    '#  b        #      ########    ###\n'+\
                    '#    b           #      #        #\n'+\
                    '#          b    d     b        b #\n'+\
                    '#              #         b       #\n'+\
                    '#      d      #    dd      b     #\n'+\
                    '#    b d     #                   #\n'+\
                    '#      d          b      b     ###\n'+\
                    '#         b     #           ######\n'+\
                    '##################################'

BANANA_ROOM = torch.zeros((len(BLOCK_TYPES),1000,1000), \
	dtype=torch.uint8)
for n in range(200**2):
	i = random.randrange(1000)
	j = random.randrange(1000)
	BANANA_ROOM[INDEX_BANANA,i,j] += 1

RAND_ROOM_WIDTH = 400

RAND_ROOM = torch.zeros((len(BLOCK_TYPES),RAND_ROOM_WIDTH,RAND_ROOM_WIDTH), \
	dtype=torch.uint8)
for n in range(RAND_ROOM_WIDTH**2//13):
	i = random.randrange(RAND_ROOM_WIDTH)
	j = random.randrange(RAND_ROOM_WIDTH)
	RAND_ROOM[INDEX_BANANA,i,j] += 1
for n in range(RAND_ROOM_WIDTH**2//50):
	i = random.randrange(RAND_ROOM_WIDTH)
	j = random.randrange(RAND_ROOM_WIDTH)
	RAND_ROOM[INDEX_DANGER,i,j] += 1
# for n in range(RAND_ROOM_WIDTH**2//20):
# 	i = random.randrange(RAND_ROOM_WIDTH)
# 	j = random.randrange(RAND_ROOM_WIDTH)
# 	RAND_ROOM[INDEX_BARRIER,i,j] += 1

# Cover with borders
for i in range(RAND_ROOM_WIDTH):
	for j in range(RAND_ROOM_WIDTH):
		if i in [0, RAND_ROOM_WIDTH-1] or j in [0, RAND_ROOM_WIDTH-1]:
			RAND_ROOM[:,i,j] = torch.zeros(len(BLOCK_TYPES))
			RAND_ROOM[INDEX_BARRIER,i,j] = 1

EMPTY_ROOM = torch.zeros((len(BLOCK_TYPES),RAND_ROOM_WIDTH,RAND_ROOM_WIDTH), \
	dtype=torch.uint8)

# Cover with borders
for i in range(RAND_ROOM_WIDTH):
	for j in range(RAND_ROOM_WIDTH):
		if i in [0, RAND_ROOM_WIDTH-1] or j in [0, RAND_ROOM_WIDTH-1]:
			EMPTY_ROOM[:,i,j] = torch.zeros(len(BLOCK_TYPES))
			EMPTY_ROOM[INDEX_BARRIER,i,j] = 1
