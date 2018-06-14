from __future__ import print_function
from __future__ import division

import global_variables as gl
import exceptions

class Monkey:
    """
    This monkey runs on a deep Q neural nework to move around collecting bananas.
    """
    def __init__(self, brain):
        """
        Initialize the monkey's state.

        Args:
            brain: A brain object.
        """
        self.food = 20
        self.dead = False
        self.pos = (0,0)
        self.food_per_banana = 6
        self.food_per_turn = 1
        self.brain = brain
        # Epsilon initialized to -1 guarantees that the policy will never
        # choose a random move.
        self.bananas = 0
        self.age = 0

    def eat(self, n):
        """
        Run this function to give the monkey a banana.

        Args:
            n: The number of bananas to give the monkey.
        """
        self.bananas += n
        self.food += self.food_per_eat*n

    def tryMove(self, s, epsilon = -1):
        """
        This passes the input vector to the Brain to evaluate the policy.

        Args:
            s: The vector that contains the information for vision.
            epsilon: Default -1. This is the chance of doing something random.

        Returns:
            0: The policy action in the form of a one-hot vector.
            1: The policy action in the form of a string.
        """
        if self.dead:
            return ' '
        # Get the policy
        a = self.brain.pi(s, epsilon)
        # Get the index that this corresponds to.
        direction_index = max(range(len(a)), key=a.__get__item)
        return a, gl.WASD[direction_index]

    def tick(self):
        """
        This function consumes food and ages the monkey.
        """
        self.food -= self.food_per_turn
        self.age += 1
        if self.food < 0:
            self.die()

    def die(self):
        """
        Kill the monkey.
        """
        self.dead = True

    def move(self, direction):
        """
        This function moves the monkey one space.

        Args:
            direction: The direction to move in terms of a 2-tensor where
                each row is one-hot.
        """
        _, direction_index = direction.max(1)
        if self.dead:
            pass
        elif direction_index  == 1:
            self.pos = (self.pos[0], self.pos[1]-1)
        elif direction_index  == 3:
            self.pos = (self.pos[0], self.pos[1]+1)
        elif direction_index  == 4:
            self.pos = (self.pos[0], self.pos[1])
        elif direction_index  == 0:
            self.pos = (self.pos[0]-1, self.pos[1])
        elif direction_index  == 2:
            self.pos = (self.pos[0]+1, self.pos[1])

    def unmove(self, direction):
        """
        This function reverses the effect of the move function.

        Args:
            direction: The direction to unmove in terms of a one-hot vector.
        """
        _, direction_index = direction.max(1)
        if self.dead:
            pass
        elif direction_index  == 3:
            self.pos = (self.pos[0], self.pos[1]-1)
        elif direction_index  == 1:
            self.pos = (self.pos[0], self.pos[1]+1)
        elif direction_index  == 4:
            self.pos = (self.pos[0], self.pos[1])
        elif direction_index  == 2:
            self.pos = (self.pos[0]-1, self.pos[1])
        elif direction_index  == 0:
            self.pos = (self.pos[0]+1, self.pos[1])


