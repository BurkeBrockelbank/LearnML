# The grid class stores information about the place where the monkey lives,
# manages input to the monkey, executes its actions, and performs a total iteration

from __future__ import print_function
from __future__ import division
import Monkey
import Roomgen
from torch import *
import torch.nn as nn
import math

SIGHT =[[0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
		[0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
		[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
		[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
		[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
		[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
		[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
		[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
		[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
		[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
		[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
		[0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
		[0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
		[0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
		[0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0]]

SIGHT =[[0,0,0,1,0,0,0],
		[0,1,1,1,1,1,0],
		[0,1,1,1,1,1,0],
		[1,1,1,1,1,1,1],
		[0,1,1,1,1,1,0],
		[0,1,1,1,1,1,0],
		[0,0,0,1,0,0,0]]

SIGHT =[[0,0,0,1,1,1,1,1,0,0,0],
		[0,0,1,1,1,1,1,1,1,0,0],
		[0,1,1,1,1,1,1,1,1,1,0],
		[1,1,1,1,1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1,1,1,1,1],
		[1,1,1,1,1,1,1,1,1,1,1],
		[0,1,1,1,1,1,1,1,1,1,0],
		[0,0,1,1,1,1,1,1,1,0,0],
		[0,0,0,1,1,1,1,1,0,0,0]]

class Grid:
	def __init__(self, monkeys, monkeyPos, room):
		self.monkeys = monkeys
		for i, p in enumerate(monkeyPos):
			self.monkeys[i].setPos(p)
		self.room = room
		self.width = len(room[0])
		self.height = len(room)

	def tick(self):
		for monkey in self.monkeys:
			# Let the monkeys see
			p = monkey.pos
			# Get the surroundings of the monkey
			surr = surroundings(p)
			# Move the monkeys
			# Check if any monkeys are eating
			# Clean up dead monkeys
			# Place bananas

	def surroundingsMap(self, pos, putMonkey=False):
		# First we need to recenter the map to just a range
		# that the monkey has a chance of seeing.
		radius = len(SIGHT)//2
		goodRows = list(range(pos[0]-radius, pos[0]+radius+1))
		goodColumns = list(range(pos[1]-radius, pos[1]+radius+1))
		surr = []
		for rr in goodRows:
			if 0 <= rr < self.height:
				thisRow = []
				for cc in goodColumns:
					if 0 <= cc < self.width:
						thisRow.append(self.room[rr][cc])
					else:
						thisRow.append(Roomgen.ABSTRACTIONS['#'])
			else:
				thisRow = [0]*len(goodColumns)
			surr.append(thisRow)
		# Now we need to block out any stuff that the monkey can't see
		# First deal with things that are too far away.
		for i, sightRow in enumerate(SIGHT):
			for j, sightEl in enumerate(sightRow):
				if sightEl == 0:
					surr[i][j] = Roomgen.ABSTRACTIONS['?']
		print('Pre barrier map:')
		print(Roomgen.concretize(surr, True))
		# Now deal with things behind barriers being hidden
		# Iterate through all the surroundings and find the
		# cones of obstruction that occur
		invisible = []
		for i, row in enumerate(surr):
			invisibleRow = []
			for j, el in enumerate(row):
				# If you hit a barrier,
				if el == Roomgen.ABSTRACTIONS['#']:
					# Eliminate all the invisible places
					invisibleRow.append(self.invisibleCone(radius, (radius, radius), (i,j)))
				else:
					invisibleRow.append([])
			invisible.append(invisibleRow)
		# Now decide which spaces need to be hidden
		for i, row in enumerate(invisible):
			for j, el in enumerate(row):
				if el != []: print('barrier at', (i,j), 'blocks', el)
				for p in el:
					if surr[p[0]][p[1]] != Roomgen.ABSTRACTIONS['#']: # We are not obscuring a barrier
						surr[p[0]][p[1]] = Roomgen.ABSTRACTIONS['?']
					elif (p[0] in [i+1,i-1]) != (p[1] in [j+1,j-1]): # We are obsuring a barrier.
						# Barriers cannot obscure barriers that are directly adjacent
						# otherwise we would run into issues like
						# #        #   ?    
						# #        #  ???     
						# # #      # ???      
						# # #      # #?        
						# #m       #m      
						# ######## #########
						# We can safely obscure the block iff it is not adjacent
						surr[p[0]][p[1]] = Roomgen.ABSTRACTIONS['?']
					else:
						print('not', p)
		# Put the monkey back
		if putMonkey:
			surr[radius][radius] = Roomgen.ABSTRACTIONS['m']
		return surr

	def surroundingVector(self, pos):
		sightVec = []
		for sightRow, surrRow in zip(SIGHT, self.surroundingsMap(pos)):
			for sightEl, surrEl in zip(sightRow, surrRow):
				if sightEl == 1: #If this block is within sight range
					# Add the character of this block
					sightVec.append(surrEl)
		# Now for each type of object, we need to have a different element space
		xMatrix = [[0]*len(Roomgen.BLOCKTYPES) for xx in len(sightVec)]
		for ii, tt in enumerate(sightVec):
			xMatrix[ii][Roomgen.ABSTRACTS.index(tt)] = 1
		xVec = []
		for row in xMatrix:
			xVec += row
		return pytorch.tensor(xVec)


	def invisibleCone(self, radius, monkeyPos, objectPos):
		# In this function we work in the reference frame where
		# the monkey is at (0,0). We need to change to this coordinate
		# system first.
		# We start our calculations with a cartesian coordinate system
		# where the object is up and to the right of the monkey (both
		# coordinates positive).
		# Not that this impolies a 90 degree rotation counter clockwise
		p = tuple([abs(tM-tO) for tO, tM in zip(objectPos, monkeyPos)])
		# Now we want to find the lines going from two corner's of the
		# monkey's space (0,1) and (1,0) to two corners of the barrier's
		# space (x,y+1) and (x+1,y)
		invisible = []
		if p[0] != 0: # The case where the barrier isn't directly above the monkey
			# l1 = lambda x: p[1]/p[0]*x+1
			# l2 = lambda x: p[1]/p[0]*(x-1)
			l1 = lambda x: (p[1]+0.5)/(p[0]-0.5)*x
			l2 = lambda x: (p[1]-0.5)/(p[0]+0.5)*x
			for x in range(p[0], radius+1):
				for y in range(max(math.ceil(l2(x)),p[1]), min(int(math.floor(l1(x))-1), radius)+1):
					if x == p[0] and y==p[1]: # Don't make the original block invisible
						pass
					else:
						invisible.append((x,y))
		else: # The case where the barrier is directly above the monkey
			for y in range(p[1]+1, radius+1):
				invisible.append((0,y))

		#print('Cone calculation from', p, 'gives invisible places', invisible)
		# Now we need to revert the coordinate system back to what it was
		# first undo the reflections imposed by the absolute value
		xMult = 1
		if objectPos[0] < monkeyPos[0]:
			xMult = -1
		yMult = 1
		if objectPos[1] < monkeyPos[1]:
			yMult = -1
		invisible = [(t[0]*xMult, t[1]*yMult) for t in invisible]
		# And now shift the origin
		invisible = [(t[0]+monkeyPos[0], t[1]+monkeyPos[1]) for t in invisible]
		return invisible

	def __repr__(self):
		toShow = list(list(x) for x in self.room)
		for monkey in self.monkeys:
			i,j = monkey.pos
			toShow[j][i] = Roomgen.ABSTRACTIONS['m']
		return Roomgen.concretize(toShow)

