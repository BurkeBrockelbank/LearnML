from __future__ import print_function
from __future__ import division

import Brain
import Roomgen

class Monkey:
    def __init__(self, brain):
        self.food = 20
        self.dead = False
        self.pos = (0,0)
        self.foodPerEat = 10
        self.foodPerTurn = 1
        self.brain = brain

    def eat(self, n):
        self.food += self.foodPerEat*n

    def tryMove(self, x):
        """
        This passes the input vector to the Brain to evaluate it.
        """
        if self.dead:
            return ' '
        # Evaluate the softmax
        softmax = self.brain(x).tolist()
        # Get the brain's instruction as the largest element
        maximum = max(softmax)
        maxIndex = softmax.index(maximum)
        directionStr = Roomgen.WASD[maxIndex]
        return directionStr

    def tick(self):
        self.food -= self.foodPerTurn
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
