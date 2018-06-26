"""
This file contains custom layers for use in monkey brains.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/layers.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable

class FoodWeight(nn.Module):
    """
        Applies a food-dependent weighting to a single channel.
    """
    def __init__(self, channels, height, width, bias = True):
        super(FoodWeight, self).init()
        self.height = height
        self.width = width
        self.channels = channels
        self.food_weight = Parameter(torch.Tensor(1))
        self.channel_weights = Parameter(torch.Tensor(1,self.channels))
        self.bias = Parameter(torch.Tensor(1))

    def reset_parameters(self):
        food_stdv = 1. / math.sqrt(self.food_weight.size(1))
        channel_stdv = 1. / math.sqrt(self.channel_weight.numel())
        self.food_weight.data.uniform_(-food_stdv, food_stdv)
        self.food_weight.data.uniform_(-channel_stdv, channel_stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-food_stdv, food_stdv)

    def forward(self, food, channel_map):
        # Weight the food and channel
        flat_channels = channel_map.view(self.channels, height*width)
        channels_weighted_flat = torch.mm(self.channel_weight, flat_channels)
        weighted_channels = channels_weighted_flat.view(channel_map.Size())
        food_weighted = food * self.food_weight
        # Add the weighted food and channel
        food_channel_sum = channels_weighted + \
            food_weighted.expand_as(channels_weighted)
        # Add in the bias
        return food_channel_sum + \
            self.bias.expand_as(food_channel_sum)

    def extra_repr(self):
        return 'height={}, width={}, food_weight={}, channel_weight={}, \
            bias={}'.format(self.height, self.width, self.food_weight, \
                self.channel_weight, self.bias is not None)




