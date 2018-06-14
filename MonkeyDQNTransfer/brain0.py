"""
This module holds brain type objects. Brain objects must have a forward
function that takes in a tensor of states and returns a one-hot direction
vector. They must have a policy function that evaluates the highest
quality move. They must have a Q function that takes in a tensor of states
and one-hot direction and returns the quality of that move.
"""

from __future__ import print_function
from __future__ import division

import torch as to
import torch.nn as nn

import global_variables as gl
import exceptions
import room_generator as rg

import random

class BrainDQN(torch.nn.Module):
    """
    This is the basic deep-Q brain class that all other brain classes will be
    based on.
    """

    def __init__(self):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        super(BrainDQN, self).__init__()

    def forward(self, s):
        Q = to.randn((1,len(gl.WASD)))
        return Q

    def argmax_action(self, s):
        Q = self.forward(s)
        maxQ, max_a = Q.max(1)
        eye = to.eye(len(gl.WASD))
        max_a_1_hot = eye[max_a]
        return max_a_1_hot

    def Q(self, s, a):
        pass

    def pi(s, epsilon):
        pass






    def __init__(self, memoryLength):
        """
        Initialize the structure of the neural net. We have the standard
        11*11*6+1=727 size for the state vector of a single turn. The input S
        will be multiple times this length to account for memory. The action
        vector is size 5. The output is of course size 1.

        To begin with, we will have three hidden layers of length 50, 30,
        and 10.
        """
        # Initialize for parent class
        super(BrainDQN, self).__init__()
        # Count the number of blocks in vision
        vision = [x.count(1) for x in Grid.SIGHT]
        vision = sum(vision)
        # Calculate input size
        inputSize = (vision*len(Roomgen.BLOCKTYPES) + 1)*memoryLength + len(Roomgen.WASD)
        self.l1 = torch.nn.Linear(inputSize, 300)
        self.l2 = torch.nn.Linear(300, 121)
        self.l3 = torch.nn.Linear(121, 5)
        self.l4 = torch.nn.Linear(5, 1)
        # Define the ReLU function
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        # Initialize weights
        torch.nn.init.xavier_uniform_(self.l1.weight.data)
        torch.nn.init.xavier_uniform_(self.l2.weight.data)
        torch.nn.init.xavier_uniform_(self.l3.weight.data)
        torch.nn.init.xavier_uniform_(self.l4.weight.data)

    def forward(self, s, a):
        """
        This calculates the reward for a given state and action.
        State is taken to be shape (1,N) and a is shape (1,N) as well
        for some N

        Args:
            s: The state vector.
            a: The action.
        """
        h = self.sigmoid(self.l1(torch.cat((s,a), -1)))
        h = self.sigmoid(self.l2(h))
        h = self.sigmoid(self.l3(h))
        h = self.l4(h)
        return h

    def maxa(self, s):
        """
        This returns the action that will maximize the quality from a,
        given state. Does not factor into gradient calculations.

        Args:
            s: The state of the system.
        Returns:
            0: The action to do.
        """
        # Turn off autograd
        with torch.no_grad():
            # Check all the possibilities for movement
            a = torch.eye(len(Roomgen.WASD))
            # Make copies of the state for each test
            sCopies = torch.empty(len(Roomgen.WASD), len(s))
            for i in range(len(sCopies)):
                sCopies[i] = s*1
            Q = self.forward(sCopies, a)
            # Maximize Q with respect to a
            maxIndex = int(Q.max(0)[1])
            # Return the action corresponding to this
            return a[maxIndex]

    def pi(self, s, epsilon, loud=False):
        """
        This is the policy for the brain that returns the action that
        maximizes the reward.
        This does not factor into the gradient calculations.

        Args:
            s: The state of the system.
            epsilon: The threshold to do a random action. Random actions are
                done a proportion of epsilon of the time.
            loud: Default False. If True, reports when the movement is random.
        returns:
            0: The action to do.
        """
        # Turn off autograd
        with torch.no_grad():
            # Check if we will be doing a random movement
            if (torch.rand(1) <= epsilon) == 1:
                # We rolled a random number less than epsilon, so we should
                # take a random move
                a = torch.eye(len(Roomgen.WASD))
                randomIndex = int(torch.randint(len(a),(1,)))
                if loud:
                    print('Random movement (',round(epsilon*100),'%)', sep='')
                return a[randomIndex]
            else:
                a = self.maxa(s)
                if loud:
                    print('Delibe movement (',round((1-epsilon)*100),\
                        '%) Q = ', self.forward(s,a).item(),sep='')
                return self.maxa(s)


