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

class MonkeyDQN(Monkey):
    """
    This is the same as the monkey class except the trymove function is
    overwritten to interface with BrainDQN.
    """
    def __init__(self, brainDQN):
        """
        Initialization is the same except we also have epsilon.
        Epsilon is stored as a local variable to be passed to 
        tryMove. I know this is bad, but this is to help it interface with
        Grid's tick function
        """
        # Epsilon initialized to -1 guarantees that the policy will never
        # choose a random move.
        self.epsilon = -1.0
        Monkey.__init__(self, brainDQN)


    def tryMove(self, s):
        """
        This passes the input vector to the Brain to evaluate the policy.

        Args:
            s: The vector that contains the information for vision.

        Returns:
            0: The policy action in the form of a string.
        """
        if self.dead:
            return ' '
        # Get the policy
        # WARNING: epsilon is loaded from a local variable here.
        a = self.brain.pi(s, self.epsilon)
        # Get the index that this corresponds to.
        directionIndex = int(a.max(0)[1])
        return Roomgen.WASD[directionIndex]