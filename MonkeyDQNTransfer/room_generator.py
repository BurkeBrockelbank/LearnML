"""
This is the room generator module. This module contains function for converting
between ASCII art maps and the maps that are used for the program (called
layer maps).

IMPORTANT NOTE: Converting a layer map into an ASCII map removes information
about multiple objects in the same grid space. Converting an ASCII map into a
layer map loses no information.
"""

from __future__ import print_function
from __future__ import division

import global_variables as gl
import torch as to
import torch.nn as nn

# Make the abstract and concretize functions
def ASCII_to_layer(ASCII_map):
    """
    Converts an ASCII map to a layer map.

    Args:
        ASCII_map: The ASCII map to convert.

    Returns:
        0: The layer map.

    Raises:
        MapSizeError: Raised if the ASCII map is not rectangular.
        SymbolError: If there are unrecognized symbols in the ASCII map.
    """
    # Split the ASCII map along newline characters.
    rows = ASCII_map.split('\n')
    # Get the height and width
    height = len(rows)
    width = len(rows[0])

    # Make sure the map is rectangular
    for row in rows:
        try:
            assert len(row) = width
        except AssertionError as e:
            raise MapSizeError('ASCII map  is not rectangular.').\
                with_traceback(e.__traceback__)

    # Initialize the layer map
    layers = to.zeros((len(gl.BLOCK_TYPES),height, width))
    # Iterate through the ASCII map
    for i, row in enumerate(rows):
        for j, symbol in enumerate(row):
            # Find the appropriate layer corresponding to this block type
            try:
                symbol_index = gl.BLOCK_TYPES.index(symbol)
            except ValueError as e:
                raise SymbolError('Symbol '+str(symbol)+' is not recognized.').\
                with_traceback(e.__traceback__)
            # Mark the layer with a 1 at the position of the block
            layers[symbol_index][i][j] = 1

    return layers

def layer_to_ASCII(layer_map,indeces=False,indexOffset=(0,0)):
    """
    This funciton converts a layer map to an ASCII map representation.

    Args:
        layer_map: The layer map in question.
        indeces: Default False. If True, show indeces in the map.
        indexOffset: Default (0,0). If indeces is True, then this offests the
            indeces in the output ASCII map.

    Returns:
        0: ASCII map string.

    Raises:
        MapSizeError: Raised if the layer map has inconsistent sizing.
    """
    # If we are asking for indeces, we will first need to call this function
    # with no optional arguments to get the basic map.
    if indeces:
        basic_map = layer_to_ASCII(layer_map)
    else:
        # Find the size of the layer map
        height = len(layer_map[0])
        width = len(layer_map[0][0])

        # Assert that the shape is correct
        try:
            assert len(layer_map) = len(gl.BLOCK_TYPES)
        except AssertionError as e:
            raise MapSizeError('Layer map has extra layers.').\
                with_traceback(e.__traceback__)
        try:
            for layer in layer_map:
                    assert len(layer) == height
                    for row in layer:
                        assert len(row) == width
        except AssertionError as e:
            raise MapSizeError('Layer map is not rectangular.').\
                with_traceback(e.__traceback__)

        # Create a blank map
        ASCII_rows = [' '*width for i in range(height)]
        # Iterate through the blank map
        for i, row in enumerate(ASCII_rows):
            for j in range(len(row)):



    mapPicture = ''
    if indeces:
        d10 = lambda x : x-x%10
        mapPicture = ' '*len(str(d10(indexOffset[0])))+'  '+str(d10(indexOffset[1]))+'+\n '+\
                    ' '*len(str(d10(indexOffset[0])))+' '
        for i in range(len(mapList[0])):
            mapPicture += str((indexOffset[1]+i)%10)
        mapPicture += '\n'
    for i, abstractRow in enumerate(mapList):
        row = ''
        for integer in abstractRow:
            row += CONCRETIZATIONS[integer]
        if indeces:
            if i==0:
                mapPicture += str(d10(indexOffset[0]))+'+'
            else:
                mapPicture += ' '*len(str(d10(indexOffset[0])))+' '
            mapPicture += str((indexOffset[0]+i)%10)
        mapPicture += row+'\n'
    mapPicture = mapPicture[:-1] #Strip the last newline character
    return mapPicture

