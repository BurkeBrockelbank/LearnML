# The grid class stores information about the place where the monkey lives,
# manages input to the monkey, executes its actions, and performs a total iteration

from __future__ import print_function
from __future__ import division
import Monkey
import Roomgen
from torch import *
import torch.nn as nn

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
		goodRows = list(range(pos[1]-radius, pos[1]+radius+1))
		goodColumns = list(range(pos[0]-radius, pos[0]+radius+1))
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
		print(Roomgen.concretize(surr))
		# Now deal with things behind barriers being hidden
		# Iterate through all the surroundings
		for j, row in enumerate(surr):
			for i, el in enumerate(row):
				# If you hit a barrier,
				if el == Roomgen.ABSTRACTIONS['#']:
					# Eliminate all the invisible places
					invisible = self.invisibleCone(radius, (radius, radius), (i,j))
					for x,y in invisible:
						surr[x][y] = Roomgen.ABSTRACTIONS['?']
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
		# system first
		p = tuple([abs(tM-tO) for tO, tM in zip(objectPos, monkeyPos)])
		# Now we want to find the lines going from two corner's of the
		# monkey's space (0,1) and (1,0) to two corners of the barrier's
		# space (x,y+1) and (x+1,y)
		invisible = []
		if p[0] != 0: # The case where the barrier isn't directly above the monkey
			l1 = lambda x: p[1]/p[0]*x+1
			l2 = lambda x: p[1]/p[0]*(x-1)
			for x in range(p[0], radius+1):
				for y in range(max(int(round(l2(x))),p[1]), min(int(round(l1(x))), radius)+1):
					if y==p[1]:
						pass
					else:
						invisible.append((x,y))
		else: # The case where the barrier is directly above the monkey
			for y in range(p[1]+1, radius+1):
				invisible.append((0,y))
		# Now we need to revert the coordinate system back to what it was
		# first undo the reflections imposed by the absolute value
		xMult = 1
		if objectPos[0] > monkeyPos[0]:
			xMult = -1
		yMult = 1
		if objectPos[1] > monkeyPos[1]:
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

