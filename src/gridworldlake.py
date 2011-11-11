#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: GRIDWORLDLAKE.PY
Date: Wednesday, January 27 2010
Description: A Gridworld with a costly lake obstacle.
"""
from gridworld import Gridworld

class GridworldLake( Gridworld ):

    def __init__( self, nrows = 8, ncols = 8 , **kwargs):

        # we have to make some assertions to make sure the we can fit
        # a lake in the middle of the gridworld

        assert  nrows > 2 and ncols > 2
        self.nstates = nrows * ncols

        self.lake = []
        for x in range(self.nstates):

            if ( 0 < (x % ncols) < ncols - 1 ) and ( ncols < x < (nrows - 1) * ncols ):
                self.lake.append(x)

        Gridworld.__init__(self, nrows = nrows, ncols = ncols, **kwargs)

    def initialize_rewards(self, a, i, j):
        """ Default reward is the final state. """

        if j == self.nstates - 1:
            return 1.0
        elif j in self.lake:
            return -1.0
        else:
            return 0.0

if __name__ == '__main__':

    gwl = GridworldLake(nrows=4,ncols = 4)
    print gwl.lake
    print gwl.rewards[0,0,5]

