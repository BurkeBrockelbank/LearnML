# This impliments some supervised training for the monkey.
#

from __future__ import print_function
from __future__ import division
import Roomgen
import Monkey
import Grid
from random import randrange


mapF = open('Map.txt', 'r')
mapStr = ''
for line in mapF:
	mapStr += line
mapF.close()

fullMap = Roomgen.abstract(mapStr)

def generateTrainingData(N):
	# This is a creator for generating data from the map
	# Create a monkey
	testMonkey = Monkey.Monkey()
	# Create the grid
	g = Grid.Grid([testMonkey], [(3,3)], fullMap)
	testData = []
	# Now go through a bunch of random places to put the monkey and ask the user which way to go
	for ii in range(N):
		blocked = True
		while blocked :
			position = (randrange(len(fullMap)), randrange(len(fullMap[0])))
			if fullMap[position[0]][position[1]] == Roomgen.ABSTRACTIONS[' ']:
				blocked = False
		# Put the monkey there
		g.monkeys[0].setPos(position)
		# Get the surroundings
		surr = g.surroundingsMap(position, True)
		# Print it out
		print('Map')
		print(Roomgen.concretize(surr, True))
		direction = input('>>>')
		if direction in list('wsad '): #Good input
			testData.append((direction, surr))
		else:
			print('Bad input')
	return testData

generateTrainingData(3)
# testMonkey = Monkey.Monkey()
# g = Grid.Grid([testMonkey], [(3,3)], fullMap)
# g.invisibleCone(5, (0,0), (3,2))