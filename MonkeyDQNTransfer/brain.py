"""
This module holds brain type objects. Brain objects must have a forward
function that takes in a tensor of states and returns a one-hot direction
vector. They must have a policy function that evaluates the highest
quality move. They must have a Q function that takes in a tensor of states
and one-hot direction and returns the quality of that move.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/brain.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

import global_variables as gl
import exceptions
import room_generator as rg

import random
import bisect

class BrainDQN(nn.Module):
    """
    This is the basic deep-Q brain class that all other brain classes will be
    based on.

    This brain has no memory implementation.

    The convention will be that every Q, s, a or other expression relating to a
    single state of phase space will have no dimension for multiples, i.e. they
    will NOT be in the form tensor([s1, s2, s3, s4, ...]) where sn is the nth
    state.

    s is a tuple (food, vision).

    food is an integer.

    vision is a 3-tensor. The first index refers to channel, the second to row, and
    the third to column.

    a is an integer 1-tensor corresponding to the
    index of the direction as specified in gl.WASD.

    Qs is a 1-tensor of length five.

    Q is a 0-tensor float corresponding to the quality of an action in a
    given state.
    """

    def __init__(self):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        super(BrainDQN, self).__init__()
        # Set the default policy
        self.pi = self.pi_epsilon_greedy

        # Initialize the neural network
        

    def forward(self, s):
        """
        Returns the 5-tensor of qualities corresponding to each direction.

        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of qualities.
        """
        # Unpack the state
        food, vision = s
        Qs = torch.randn(1,len(gl.WASD))
        return Qs

    def pi_greedy(self, s):
        """
        Gives the quality and action corresponding to a totally greedy policy.

        Args:
            s: The state of the system.

        Returns:
            0: The maximal quality.
            1: The action that maximizes quality.
        """
        Qs = self.forward(s)
        max_Q, max_a = Qs.max(0)
        return max_Q, max_a

    def argmax_a(self, s):
        """
        Returns the action that maximizes quality for the state s.
        
        Args:
            s: The state of the system.

        Returns:
            0: Action which maximizes the quality of the state.
        """
        max_Q, max_a = self.pi_greedy(s)
        return max_a

    def Q(self, s, a):
        """
        Returns the quality of performing action a in state s.

        Args:
            s: The state of the system.
            a: The action.

        Returns:
            0: The quality of that move.
        """
        Qs = self.forward(s)
        return Qs[a]


    def pi_epsilon_greedy(self, s, epsilon):
        """
        Enacts the epsilon-greedy policy of the brain.

        Args:
            s: The state of the system.
            epsilon: The probibility of taking a random movement.

        Returns:
            0: The determined quality of this move.
            1: The action the policy points to.
        """
        # Roll a random number
        if random.uniform(0, 1) < epsilon:
            # Make a random movement
            a = random.randrange(0,len(gl.WASD))
            # Calculate the qualities
            return self.Q(s,a), a
        else:
            # Just take the best action
            return self.pi_greedy(s)

    def pi_probabilistic(self, s):
        """
        Enacts the policy of qualities corresponding to probabilities.
        Qualities are fed into a softmax and then an action selected
        according to those probabilities.

        Args:
            s: The gamesate.

        Returns:
            0: The quality of the action.
            1: The action to be taken.
        """
        # Find the qualities.
        Qs = self.forward(s)
        # Run this through a softmax
        Q_softmax = F.softmax(Qs)
        # Calculate the CDF.
        CDF = [Q_softmax[0]]
        # The CDF is one element shorter than the probabilities because the
        # last element is one.
        for p in Q_softmax[1:-1]:
            CDF.append(CDF[-1]+p)
        # Generate a random number in a uniform distribution from 0 to 1.
        roll = random.uniform(0, 1)
        # Get the action this corresponds to.
        a = bisect.bisect(CDF, roll)
        # Return the quality and action.
        return Qs[a], a