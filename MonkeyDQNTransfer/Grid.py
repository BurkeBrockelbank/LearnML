"""
This is the grid module, which contains the grid class. Grid classes manage the
game state.

Project: Monkey Deep Q Recurrent Network with Transfer Learning
Path: root/grid.py
"""

from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import global_variables as gl
import exceptions
import room_generator as rg

import math
import random
import copy

class Grid:
    """
    The grid object keeps track of and alters the game state. All queries
    actions related to the game state should go through here.

    Eventually this object will support multiple monkeys. Efforts are made to
    enforce this capability but as of now, this object is only tested for a
    single monkey.
    """
    def __init__(self, monkeys, channel_map, place_monkeys=True):
        """
        Initialization for the grid object.

        Args:
            monkeys: A list of monkeys.
            room: A channel map. Data type is unit8.
            place_monkeys: Default True. If True, monkeys are assumed to be
            in the channel map already. If False, monkeys are placed into the
                channel map from their positions.
        """
        self.monkeys = monkeys
        self.channel_map = channel_map
        self.width = len(channel_map[0])
        self.height = len(channel_map)
        self.turn_count = 0
        # Add in the monkeys into the room
        monkey_channel_index = gl.BLOCK_TYPES.index('m')
        for monkey in self.monkeys:
            i,j = monkey.pos
            self.channel_map[monkey_channel_index][i][j] += 1

    def tick(self, control, directions = [], invincible = False, loud=True):
        """
        This function moves the entire grid and all the monkeys forward one
        timestep.

        Args:
            control: If 0, the monkey is queried on its moves. If 1,
                a list of movement directions must be given which is the same
                length as self.monkeys. If 2, the user is queried for a
                movment direction.
            directions: A list of actions. One for each monkey. Only applies if
                control is 1.
            invincible: Default False. If true, the monkey is not removed if it
                dies.

        Raises:
            ControlError: Raised if control is not properly defined.
        """
        # Report the turn if set to loud
        if loud:
            print('TURN', self.turn_count)

        # Instantiate a list for dying monkeys
        dead_monkeys = []
        surrs = []
        actions = []

        # Iterate through all the monkeys
        for monkey_index, monkey in enumerate(self.monkeys):
            # Get the surroundings of the monkey.
            surr = self.surroundings(monkey.pos)
            # Print details for this tick
            if loud:
                # Print monkey number and number of bananas
                print('Monkey', monkey_index, 'bananas', monkey.food, 'age', monkey.age)
                # Get the ascii map
                text_map = rg.channel_to_ASCII(surr,indeces=True,index_offset=monkey.pos)
                # Print the ascii map
                print(text_map)

            # Determine control type and get action
            if control == 2:
                # Get user input.
                # Loop until good input is given
                need_input = True
                while need_input:
                    # Get action
                    action_string = input('>>>')
                    # Turn action string into an action integer
                    try:
                        action = gl.WASD.index(action_string)
                        need_input = False
                    except ValueError:
                        print('Input must be w, a, s, d, or space.')
            elif control == 1:
                # Get input from a list of directions
                try:
                    action = directions[monkey_index]
                except IndexError as e:
                    raise ControlError('Directions not specified').\
                        with_traceback(e.__traceback__)
                if action not in range(len(gl.WASD)):
                    raise ControlError('Action ' + str(action) + \
                        ' for monkey ' + str(monkey_index)+' is not valid.')
                # Print out the action if loud is on
                if loud:
                    # Get the string action
                    action_string = gl.WASD[action]
                    input('>>>'+action_string)
            elif control == 0:
                # Get action from monkey's brain
                action = monkey.action(surr)
                # Print out the action if loud is on
                if loud:
                    # Get the string action
                    action_string = gl.WASD[action]
                    input('>>>'+action_string)
            else:
                raise ControlError('Control must be specified as 0, 1, or 2')

            # Add the surroundings and actions to the record.
            surrs.append(surr)
            actions.append(action)

            # Now we want to move the monkey
            monkey.move(action)
            # Get the blocks on this space
            this_space = self.channel_map[:][monkey.pos[0]][monkey.pos[1]]
            # Check if the monkey is trying to move to a barrier
            if this_space[gl.BLOCK_TYPES.index('#')] >= 1:
                # Need to unmove the monkey.
                monkey.unmove(action)
                # Get the blocks on this space
                this_space = self.channel_map[:][monkey.pos[0]][monkey.pos[1]]
            # Feed the monkey any bananas on this spot
            for i in range(this_space[gl.BLOCK_TYPES.index('b')]):
                monkey.eat()
            # Check if the monkey is in danger
            if this_space[gl.BLOCK_TYPES.index('d')] >= 1:
                monkey.dead = True
            # The monkey now ages and consumes food.
            monkey.tick()
            # Check if the monkey is starving to death.
            if monkey.food <= 0:
                monkey.dead = True
            # Check if the monkey is invincible
            if invincible:
                monkey.dead = False
            # Clean up the monkey if need be
            if monkey.dead:
                # Mark monkey for cleanup
                dead_monkeys.append(monkey_index)
        # Remove dead monkeys
        for dead_index in dead_monkeys[::-1]:
            del monkeys[dead_index]
        # Check if there are any monkeys left
        if len(self.monkeys) == 0:
            print('All monkeys have died.')

        return surrs, actions


    def surroundings(self, pos):
        """
        This function finds the surroundings of a monkey based on its
        sightlines. The sightline is assumed to be a square matrix.

        Args:
            pos: The position (integer couple) around wich we will center the
                map.

        Returns:
            0: A cropped channel map that has been obscured according to
                gl.SIGHT.
        """
        # The first thing to do is pad the channel map with enough zeros that
        # we could put the monkey anywhere and still slice the array
        radius = len(gl.SIGHT)//2
        padded = torch.zeros(self.channel_map.size(),dtype=torch.uint8)
        for i in range(len(gl.index)):
            # Pad the barrier channel with ones and everything else with zeros.
            padding_value = 0
            if i == gl.index('#'):
                padding_value = 1
            padded[i] = F.pad(self.channel_map[i],\
            (radius, radius, radius, radius), value=padding_value)
        # Note: The elements of padded do not share pointers with the elements
        # of self.channel_map.
        # Slice the array
        sliced = padded[pos[0]:pos[0]+len(gl.SIGHT), \
            pos[1]:pos[1]+len(gl.SIGHT)]
        # Now we need to obscure any blocks that are deemed invisible.
        for i, row in enumerate(gl.SIGHT):
            for j, el in enumerate(row):
                # An invisible block is marked as a zero in SIGHT
                if el == 0:
                    # Turn this spot into a barrier
                    for k in range(len(gl.sight)):
                        if k == gl.index('#'):
                            sliced[k][i][j] = 1
                        else:
                            sliced[k][i][j] = 0
        # Return the surroundings
        return sliced


    def __str__(self):
        """
        This function returns the ASCII map.

        Returns:
            0: ASCII string.
        """
        return rg.channel_to_ASCII(self.channel_map)

    def __repr__(self):
        """
        This function returns the channel maps in a string plus the
        list of monkeys in strings.

        Returns:
            0: Representation string.
        """
        return repr(self.turn_count) + repr(self.channel_map) + \
            str(self.monkeys)





