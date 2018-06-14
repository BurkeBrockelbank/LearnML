# The grid class stores information about the place where the monkey lives,
# manages input to the monkey, executes its actions, and performs a total iteration

from __future__ import print_function
from __future__ import division
import Monkey
import Roomgen
import Exceptions
import torch
import math
from random import randrange

# SIGHT =[[0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0],
#         [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
#         [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
#         [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
#         [0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0],
#         [0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0]]

# SIGHT =[[0,0,0,1,0,0,0],
#         [0,1,1,1,1,1,0],
#         [0,1,1,1,1,1,0],
#         [1,1,1,1,1,1,1],
#         [0,1,1,1,1,1,0],
#         [0,1,1,1,1,1,0],
#         [0,0,0,1,0,0,0]]

# SIGHT =[[0,0,0,1,1,1,1,1,0,0,0],
#         [0,0,1,1,1,1,1,1,1,0,0],
#         [0,1,1,1,1,1,1,1,1,1,0],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [1,1,1,1,1,1,1,1,1,1,1],
#         [0,1,1,1,1,1,1,1,1,1,0],
#         [0,0,1,1,1,1,1,1,1,0,0],
#         [0,0,0,1,1,1,1,1,0,0,0]]
SIGHT = [[1]*11 for _ in range(11)]

class Grid:
    def __init__(self, monkeys, monkeyPos, room):
        self.monkeys = monkeys
        for i, p in enumerate(monkeyPos):
            self.monkeys[i].setPos(p)
        self.room = room
        self.width = len(room[0])
        self.height = len(room)
        self.turnCount = 0

    def tick(self, trainingData = False, control='manual', wait = False,
        directions=[], quiet = False, invincible = False):
        """
        Ticking function for a grid object moves all the monkeys, distributes
        bananas, and kills monkeys if it needs to.

        Args:
            control: Must be either 'user', 'manual', or 'auto'. If control is 'manual',
                directions must be populated with string directions of the same length
                as the number of monkeys.
            directions: The directions monkeys should go in if control is manual.
            invincible: Default False. If true, monkeys are not removed after death.

        Raises:
            ControlError: Thrown in the case that the control argument is not valid.
        """
        if not quiet:
            print('TURN', self.turnCount)
        for n, monkey in enumerate(self.monkeys):
            if not quiet:
                print('M',n,' B',monkey.food,sep='')
            # Let the monkeys see
            p = monkey.pos
            surrVec, surrMap = self.surroundingVector(p, putMonkey=True)
            # Print the surroundingsMap
            if not quiet:
                print(Roomgen.concretize(surrMap,indeces=True,indexOffset=\
                    (p[0]-len(SIGHT)//2,p[1]-len(SIGHT)//2)))
            if control == 'user':
                # Ask the user for their input
                needInput = True
                while needInput:
                    direction = input('>>>')
                    if direction in list(Roomgen.WASD):
                        needInput = False
                    else:
                        print('Input must be w, a, s, d, or space.')
            elif control == 'auto':
                # Automatic control
                x = torch.cat((torch.FloatTensor([monkey.food]),surrVec))
                x = torch.tensor(x)
                direction = monkey.tryMove(x)
                if not quiet:
                    if wait:
                        input('>>>'+direction)
                    else:
                        print('>>>'+direction)
            elif control == 'manual':
                try:
                    direction = directions[n]
                except IndexError:
                    raise Exceptions.ControlError('Directions were not given to monkey ' +str(n))
                if not quiet:
                    if wait:
                        input('>>>'+direction)
                    else:
                        print('>>>'+direction)
            else:
                raise Exception.ControlError('Argument control must be in [\'user\', \'manual\', '+\
                                            '\'auto\'] but was given as,' + str(control))
            # We have the direction, so check if the monkey can move that way
            monkey.move(direction)
            if self.room[monkey.pos[0]][monkey.pos[1]] == Roomgen.ABSTRACTIONS['#']:
                # This was a bad move
                monkey.unmove(direction)
            # Check if any monkeys are eating
            if self.room[monkey.pos[0]][monkey.pos[1]] == Roomgen.ABSTRACTIONS['b']:
                self.room[monkey.pos[0]][monkey.pos[1]] = Roomgen.ABSTRACTIONS[' ']
                bananasToPlace = 1
                monkey.eat(bananasToPlace)
                # Replace the bananas
                while bananasToPlace > 0:
                    bi = randrange(len(self.room))
                    bj = randrange(len(self.room[0]))
                    if self.room[bi][bj] == Roomgen.ABSTRACTIONS[' ']:
                        self.room[bi][bj] = Roomgen.ABSTRACTIONS['b']
                        bananasToPlace -= 1
            # Check if monkey fell in lava
            if self.room[monkey.pos[0]][monkey.pos[1]] == Roomgen.ABSTRACTIONS['d']:
                monkey.die()
            # The monkey now expends food if it has it
            monkey.tick()
            # Clean up dead monkies
            if monkey.dead and not invincible:
                print('Monkey', n,'is dead')
                del self.monkeys[n]
        self.turnCount += 1
        # Check if any living monkeys are left
        if len(self.monkeys) == 0:
            print('All monkeys are dead')
            raise Exceptions.DeathError
        return direction, surrVec

    def surroundingsMap(self, pos, putMonkey=False):
        # First we need to recenter the map to just a range
        # that the monkey has a chance of seeing.
        radius = len(SIGHT)//2
        goodRows = list(range(pos[0]-radius, pos[0]+radius+1))
        goodColumns = list(range(pos[1]-radius, pos[1]+radius+1))
        surr = []
        for rr in goodRows:
            if 0 <= rr < self.height:
                thisRow = []
                for cc in goodColumns:
                    if 0 <= cc < self.width:
                        thisEl = self.room[rr][cc]
                        # We have to check if there is a monkey here
                        if thisEl == Roomgen.ABSTRACTIONS[' ']: #Empty space can hold monkeys
                            for monkey in self.monkeys:
                                if monkey.pos == (rr,cc): # There is a monkey here
                                    thisEl = Roomgen.ABSTRACTIONS['m']
                        thisRow.append(thisEl)
                    else:
                        thisRow.append(Roomgen.ABSTRACTIONS['#'])
            else:
                thisRow = [0]*len(goodColumns)
            surr.append(thisRow)
        # Now we need to block out any stuff that the monkey can't see
        # First deal with things that are too far away.
        for i, sightRow in enumerate(SIGHT):
            for j, sightEl in enumerate(sightRow):
                if sightEl == 0:
                    surr[i][j] = Roomgen.ABSTRACTIONS['?']
        # # Now deal with things behind barriers being hidden
        # # Iterate through all the surroundings and find the
        # # cones of obstruction that occur
        # invisible = []
        # for i, row in enumerate(surr):
        #     invisibleRow = []
        #     for j, el in enumerate(row):
        #         # If you hit a barrier,
        #         if el == Roomgen.ABSTRACTIONS['#']:
        #             # Eliminate all the invisible places
        #             invisibleRow.append(self.invisibleCone(radius, (radius, radius), (i,j)))
        #         else:
        #             invisibleRow.append([])
        #     invisible.append(invisibleRow)
        # # Now decide which spaces need to be hidden
        # hiddenSpots = []
        # for i, row in enumerate(invisible):
        #     for j, el in enumerate(row):
        #         for p in el:
        #             if p not in hiddenSpots:
        #                 if surr[p[0]][p[1]] != Roomgen.ABSTRACTIONS['#']: # We are not obscuring a barrier
        #                     hiddenSpots.append(p)
        #                 elif (p[0] in [i+1,i-1]) != (p[1] in [j+1,j-1]): # We are obscuring a barrier.
        #                     # Barriers cannot normally obscure barriers that are directly adjacent
        #                     # otherwise we would run into issues like
        #                     # #        #   ?    
        #                     # #        #  ???     
        #                     # # #      # ???      
        #                     # # #      # #?        
        #                     # #m       #m      
        #                     # ######## #########
        #                     # However this leaves us with a few cases which still need to be fixed.
        #                     # For example,
        #                     # m ## -->    m ## instead of m #?
        #                     # We will obscure only if there is also a barrier adjacent in the other
        #                     # direction or it is directly in line with the monkey
        #                     pDiff = tuple([a-b for a,b in zip(p,(i,j))])
        #                     mDiff = tuple([a-b for a,b in zip(pos,(i,j))])
        #                     if pDiff in [(0,1), (0,-1)]:
        #                         # Like m
        #                         # or   m   ##
        #                         #      m
        #                         # Find out which direction the monkey is (monkey is at j=radius)
        #                         if i > radius:
        #                             # Monkey is above
        #                             if surr[p[0]-1][p[1]] == Roomgen.ABSTRACTIONS['#']:
        #                                 # Obscure it
        #                                 hiddenSpots.append(p)
        #                         elif i == radius:
        #                             # Monkey is beside, so obscure it
        #                             hiddenSpots.append(p)
        #                         else:
        #                             # Monkey is below
        #                             if surr[p[0]+1][p[1]] == Roomgen.ABSTRACTIONS['#']:
        #                                 # Obscure it
        #                                 hiddenSpots.append(p)
        #                     else: # pDiff in [(1,0), (-1,0)]
        #                         # Like mmm
        #                         #       #
        #                         #       #
        #                         # Find out which direction the monkey is
        #                         if j > radius:
        #                             # Monkey is to left
        #                             if surr[p[0]][p[1]-1] == Roomgen.ABSTRACTIONS['#']:
        #                                 # Obscure it
        #                                 hiddenSpots.append(p)
        #                         elif j == radius:
        #                             # Monkey is above/below, so obscure it
        #                             hiddenSpots.append(p)
        #                         else:
        #                             # Monkey is to right
        #                             if surr[p[0]][p[1]+1] == Roomgen.ABSTRACTIONS['#']:
        #                                 # Obscure it
        #                                 hiddenSpots.append(p)
        #                 else: # Obscuring a barrier that is not adjacent (always fine)
        #                     hiddenSpots.append(p)
        # for p in hiddenSpots:
        #     surr[p[0]][p[1]] = Roomgen.ABSTRACTIONS['?']
        # Put the monkey back
        if putMonkey:
            surr[radius][radius] = Roomgen.ABSTRACTIONS['m']
        return surr

    def surroundingVector(self, pos, putMonkey = False):
        sightVec = []
        surroundingMap = self.surroundingsMap(pos, putMonkey)
        for sightRow, surrRow in zip(SIGHT, surroundingMap):
            for sightEl, surrEl in zip(sightRow, surrRow):
                if sightEl == 1: #If this block is within sight range
                    # Add the character of this block
                    sightVec.append(surrEl)
        # Now for each type of object, we need to have a different element space
        xMatrix = [[0]*len(Roomgen.BLOCKTYPES) for xx in sightVec]
        for ii, tt in enumerate(sightVec):
            xMatrix[ii][Roomgen.ABSTRACTS.index(tt)] = 1
        xVec = []
        for row in xMatrix:
            xVec += row
        return torch.tensor(xVec), surroundingMap


    def invisibleCone(self, radius, monkeyPos, objectPos):
        # In this function we work in the reference frame where
        # the monkey is at (0,0). We need to change to this coordinate
        # system first.
        # We start our calculations with a cartesian coordinate system
        # where the object is up and to the right of the monkey (both
        # coordinates positive).
        # Not that this impolies a 90 degree rotation counter clockwise
        p = tuple([abs(tM-tO) for tO, tM in zip(objectPos, monkeyPos)])
        # Now we want to find the lines going from two corner's of the
        # monkey's space (0,1) and (1,0) to two corners of the barrier's
        # space (x,y+1) and (x+1,y)
        invisible = []
        # The case where the barrier is directly to the right of the monkey
        if p[1] == 0:
            for x in range(p[0]+1, radius+1):
                invisible.append((x,0))
        # The case where the barrier is directly above the monkey
        elif p[0] == 0:
            for y in range(p[1]+1, radius+1):
                invisible.append((0,y))
        else: # The case where the barrier isn't directly above the monkey
            # l1 = lambda x: p[1]/p[0]*x+1
            # l2 = lambda x: p[1]/p[0]*(x-1)
            l1 = lambda x: (p[1]+0.5)/(p[0]-0.5)*x
            l2 = lambda x: (p[1]-0.5)/(p[0]+0.5)*x
            for x in range(p[0], radius+1):
                for y in range(max(math.ceil(l2(x)),p[1]), min(math.floor(l1(x)), radius)+1):
                    if x == p[0] and y==p[1]: # Don't make the original block invisible
                        pass
                    else:
                        invisible.append((x,y))

        #print('Cone calculation from', p, 'gives invisible places', invisible)
        # Now we need to revert the coordinate system back to what it was
        # first undo the reflections imposed by the absolute value
        xMult = 1
        if objectPos[0] < monkeyPos[0]:
            xMult = -1
        yMult = 1
        if objectPos[1] < monkeyPos[1]:
            yMult = -1
        invisible = [(t[0]*xMult, t[1]*yMult) for t in invisible]
        # And now shift the origin
        invisible = [(t[0]+monkeyPos[0], t[1]+monkeyPos[1]) for t in invisible]
        return invisible

    def __repr__(self):
        toShow = list(list(x) for x in self.room)
        for monkey in self.monkeys:
            i,j = monkey.pos
            toShow[j][i] = Roomgen.ABSTRACTIONS['m']
        return Roomgen.concretize(toShow)

