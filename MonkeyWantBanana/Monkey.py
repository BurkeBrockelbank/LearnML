from __future__ import print_function
from __future__ import division

class Monkey:
	def __init__(self):
		self.food = 20
		self.dead = False
		self.pos = (0,0)

	def eat(self, n):
		self.food += 10*n

	def tryMove(self):
		# This is the function that will eventually
		# do all the computation for moving.
		# Right now it just goes to the left to the left.
		# Tick all resources
		self.tick()
		if self.dead:
			return '0'
		return 'l'

	def tick(self):
		self.food -= 1
		if self.food < 0:
			self.die()

	def see(self, map):
		pass

	def die(self):
		self.dead = True

	def setPos(self, p):
		self.pos = p
	def move(self, direction):
		if self.dead:
			pass
		elif direction == 'a':
			self.pos = (self.pos[0], self.pos[1]-1)
		elif direction == 'd':
			self.pos = (self.pos[0], self.pos[1]+1)
		elif direction == ' ':
			self.pos = (self.pos[0], self.pos[1])
		elif direction == 'w':
			self.pos = (self.pos[0]-1, self.pos[1])
		elif direction == 's':
			self.pos = (self.pos[0]+1, self.pos[1])
	def unmove(self, direction):
		if self.dead:
			pass
		elif direction == 'd':
			self.pos = (self.pos[0], self.pos[1]-1)
		elif direction == 'a':
			self.pos = (self.pos[0], self.pos[1]+1)
		elif direction == ' ':
			self.pos = (self.pos[0], self.pos[1])
		elif direction == 's':
			self.pos = (self.pos[0]-1, self.pos[1])
		elif direction == 'w':
			self.pos = (self.pos[0]+1, self.pos[1])
