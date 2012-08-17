#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: GRIDWORLD.PY
Date: Monday, January 11 2010
Description: A class for building gridworlds.

Note: This should be considered deprecated. Use gridworld8.py (which will eventually become gridworld.py).
"""

import os, sys, getopt, pdb, string
import random as pr
import numpy as np
from markovdp import MDP, SparseMDP
import scipy.cluster.vq as vq

class SparseGridworld( SparseMDP ):
    def __init__(self, nrows = 8, ncols = 8):
        self.nrows = nrows
        self.ncols = ncols
        self.nstates = nrows * ncols
        self.nactions = 4

        self.left_edge = []
        self.right_edge = []
        self.top_edge = []
        self.bottom_edge = []
        self.gamma = 0.9

        for x in range(self.nstates):

            # note that edges are not disjoint, so we cannot use elif

            if x % self.ncols == 0:
                self.left_edge.append(x)

            if 0 <= x < self.ncols:
                self.top_edge.append(x)

            if x % self.ncols == self.ncols - 1:
                self.right_edge.append(x)

            if (self.nrows - 1) * self.ncols <= x <= self.nstates:
                self.bottom_edge.append(x)

        SparseMDP.__init__(self, nstates = self.nrows * self.ncols, nactions = 4)


    def coords(self, s):
        return s / self.ncols, s % self.ncols

    def initialize_rewards(self):
        """ Default reward is the final state. """

        r = np.zeros(self.nstates)
        r[-1] = 1.0
        return r

    def initialize_model(self, a, i):
        """
        Simple gridworlds assume four actions -- one for each cardinal direction.
        """

        if a == 0:
            if i in self.left_edge:
                return [(i,1.0)]
            else:
                return [(i-1, 1.0)]
        elif a == 1:
            if i in self.top_edge:
                return [(i,1.0)]
            else:
                return [(i-self.ncols,1.0)]
        elif a == 2:
            if i in self.right_edge:
                return [(i,1.0)]
            else:
                return [(i+1,1.0)]
        elif a == 3:
            if i in self.bottom_edge:
                return [(i,1.0)]
            else:
                return [(i + self.ncols,1.0)]


class Gridworld( MDP ):
    """
    This is a rather unfancy gridworld that extends the basic discrete
    MDP framework to parameterize the size of the gridworld. Subclass
    this for more advanced gridworlds with "walls" etc.

    A good way to add obstacles is to define a set of indices with a
    good descriptive name, and then deal with those special cases in
    the initialization functions for the transition and reward
    models. See for example the way the boundaries are delt with.
    """

    def __init__(self, nrows = 8, ncols = 8):
        self.nrows = nrows
        self.ncols = ncols
        self.nstates = nrows * ncols
        self.nactions = 4

        self.left_edge = []
        self.right_edge = []
        self.top_edge = []
        self.bottom_edge = []
        self.gamma = 0.9

        for x in range(self.nstates):

            # note that edges are not disjoint, so we cannot use elif

            if x % self.ncols == 0:
                self.left_edge.append(x)

            if 0 <= x < self.ncols:
                self.top_edge.append(x)

            if x % self.ncols == self.ncols - 1:
                self.right_edge.append(x)

            if (self.nrows - 1) * self.ncols <= x <= self.nstates:
                self.bottom_edge.append(x)

        MDP.__init__(self, nstates = self.nrows * self.ncols, nactions = 4)


    def coords(self, s):
        return s / self.ncols, s % self.ncols

    def initialize_rewards(self, a, i, j):
        """ Default reward is the final state. """

        if j == self.nstates - 1:
            return 1.0
        else:
            return 0.0

    def initialize_model(self, a, i, j):
        """
        Simple gridworlds assume four actions -- one for each cardinal direction.
        """

        if a == 0:
            # left

            if i in self.left_edge:
                if i == j:
                    return 1.0
                else:
                    return 0.0

            elif j == i - 1:
                return 1.0

            else:
                return 0.0

        elif a == 1:
            # up

            if i in self.top_edge:
                if i == j:
                    return 1.0
                else:
                    return 0.0

            elif j == i - self.ncols:
                return 1.0

            else:
                return 0.0

        elif a == 2:
            # right

            if i in self.right_edge:
                if i == j:
                    return 1.0
                else:
                    return 0.0

            elif j == i + 1:
                return 1.0

            else:
                return 0.0

        elif a == 3:
            # down

            if i in self.bottom_edge:
                if i == j:
                    return 1.0
                else:
                    return 0.0

            elif j == i + self.ncols:
                return 1.0

            else:
                return 0.0

if __name__ == '__main__':

    gw = Gridworld()

    t = gw.trace(10000)
    states = [x[0] for x in t]
    rewards = [x[2] for x in t]

    # sanity check the distribution over visited states
    print np.histogram(states, bins=range(gw.nstates))
    print np.histogram(rewards, bins = [0,1,2])

    gws = SparseGridworld(nrows = 32, ncols = 64) # without a sparse rep. this would blowup
    t = gws.trace(10000)

