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
import layers

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

    def forward(self, s):
        """
        Returns the 5-tensor of qualities corresponding to each direction.

        Args:
            s: The state of the system.
        
        Returns:
            0: 5-tensor of qualities.

        Raises:
            BrainError: This is only meant to be a parent class. Using this class
            as if it were functional should raise an error.
        """
        raise exceptions.BrainError

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
        Q_softmax = F.softmax(Qs, dim=0)
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
        return Qs[a], a, Q_softmax[a]


class BrainLinear(BrainDQN):
    """
    This implements the naive approach of the monkey brain as described in the
    presentation. It is a simple linear model.

    This brain has no memory implementation.
    """

    def __init__(self):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        BrainDQN.__init__(self)
        # Set the default policy
        self.pi = self.pi_probabilistic

        # Initialize the neural network
        # We just need to take the input size:
        # 1 [food] + 11*11 [grid size] * 4 [number of channels] = 485
        self.layer = nn.Linear(485,5)

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
        
        # First we need to flatten out the state and add in the food.
        vision_float = vision.float().view(-1)
        food_float = torch.FloatTensor([food])
        state = torch.cat((food_float,vision_float),0)

        # Secondly we need to run this though the linear layer.
        Qs = self.layer(state)

        return Qs


class BrainV1(BrainDQN):
    """
    This implements the first approach of the monkey brain as described in the
    presentation.

    This brain has no memory implementation.
    """

    def __init__(self):
        """
        Initialize the architecture of the neural net.
        """
        # Initialize the parent class
        BrainDQN.__init__(self)
        # Set the default policy
        self.pi = self.pi_epsilon_greedy

        # Initialize the neural network
        # Part 1: Desire Mapping
        self.DM_weight = layers.FoodWeight(3, len(gl.SIGHT), len(gl.SIGHT[0]))
        self.DM_conv = nn.Conv2d(3, 1, 3, stride=2)
        # Part 2: Path Finding
        self.PF_conv1 = nn.Conv2d(2, 4, 3)
        self.PF_conv2 = nn.Conv2d(4, 1, 5)
        # Part 3: Evaluation
        self.EV_linear1 = nn.Linear(50,8)
        self.EV_linear2 = nn.Linear(8,5)

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
        
        # Part 1: Desire Mapping
        # Get the correct input channels.
        DM_channels = vision.index_select(0,torch.tensor(\
            [gl.INDEX_MONKEY, gl.INDEX_BANANA, gl.INDEX_DANGER]))
        # Weight the channels
        DM_weighted = self.DM_weight(food, DM_channels)
        DM_weighted = F.relu(DM_weighted)
        # Build desire map
        # Need to add another dimension with None indexing
        DM_desire_map = self.DM_conv(DM_weighted[None])

        # Part 2: Path Finding
        # Get the correct input channels
        PF_channels = vision.index_select(0,torch.tensor(\
            [gl.INDEX_BARRIER, gl.INDEX_DANGER]))
        # Find walls
        PF_walls = self.PF_conv1(PF_channels[None].type(torch.FloatTensor))
        PF_walls = F.sigmoid(PF_walls)
        # Determine passability
        PF_path_map = self.PF_conv2(PF_walls)
        PF_path_map = F.relu(PF_path_map)

        # Part 3： Evaluation
        # Flatten
        EV_flat = torch.cat((DM_desire_map.view(-1), PF_path_map.view(-1)), 0)
        # Fully connected ReLU layers
        h = F.relu(self.EV_linear1(EV_flat))
        Qs = F.relu(self.EV_linear2(h))

        return Qs