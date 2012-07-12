#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: GRIDWORLD.PY
Date: Monday, January 11 2010
Description: A class for building gridworlds.
"""

import os, sys, getopt, pdb, string
import random as pr
import numpy as np
import numpy.linalg as la
from markovdp import MDP,FastMDP,SparseMDP

class SparseGridworld8( SparseMDP ):

    def __init__(self, nrows = 5, ncols = 5, walls=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)], endstates = [0], nrbf = 15):
        self.nrows = nrows
        self.ncols = ncols

        self.walls = walls
        grid = [self.coords(i) for i in range(self.nrows * self.ncols)]
        grid = [s for s in grid if not s in self.walls]
        self.states = dict([(i,s) for (i,s) in enumerate(grid)])
        self.rstates = dict([(s,i) for (i,s) in enumerate(grid)]) # reverse lookup by grid coords

        self.nstates = len(self.states)
        self.nactions = 8
        self.endstates = endstates

        SparseMDP.__init__(self, nstates = self.nstates, nactions = self.nactions)

        self.rbf_loc = [np.array(self.coords(i)) for i in pr.sample(self.sindices, nrbf)]

    def is_state(self, s):
        return s in self.rstates

    def state_index(self, s):
        return self.rstates[s]

    def coords(self, s):
        return (s / self.ncols, s % self.ncols)

    def dcoords(self, a):
        # turn action index into an action A st. CS + A = NS
        actions = [np.array((-1,0)),np.array((-1,-1)),
                   np.array((0,-1)),np.array((1,-1)),
                   np.array((1,0)), np.array((1,1)),
                   np.array((0,1)), np.array((-1,1))]

        return actions[a]

    def phi(self, s, a):
        r = np.zeros(len(self.rbf_loc) * self.nactions + 1)

        # compute the rbf functions
        j = np.array(self.coords(s))
        
        for i,c in enumerate(self.rbf_loc):
            #print la.norm(c - j) ** 2, -.0001 *la.norm(c - j,ord=np.inf) ** 2, np.exp(-.001 *la.norm(c - j,ord=np.inf) ** 2)
            r[a * len(self.rbf_loc) + i] = np.exp(-.001 *la.norm(c - j, ord=np.inf) ** 2)

        r[-1] = 1.0
        return r


    def nfeatures(self):
        return len(self.rbf_loc) * self.nactions + 1

    # initialize_* methods create a complete model -- inefficient
    def initialize_rewards(self):

        r = np.zeros(self.nstates)
        for es in self.endstates:
            r[es] = 1.0
        return r

    def initialize_model(self, a, i):

        cs = self.states[i]
        ac = self.dcoords(a)
        ns = cs + ac
        ns = tuple(ns)

        if self.is_state(ns):
            return [(self.state_index(ns),1.0)]
        else:
            return [(i,1.0)]

class FastGridworld8( FastMDP ):

    def __init__(self, nrows = 5, ncols = 5, endstates=[0], walls=[]):
        self.nrows = nrows
        self.ncols = ncols

        self.walls = walls
        grid = [self.coords(i) for i in range(self.nrows * self.ncols)]
        grid = [s for s in grid if not s in self.walls]
        self.states = dict([(i,s) for (i,s) in enumerate(grid)])
        self.rstates = dict([(s,i) for (i,s) in enumerate(grid)]) # reverse lookup by grid coords

        self.nstates = len(self.states)
        self.nactions = 8
        self.endstates = endstates

        FastMDP.__init__(self, nstates = self.nstates, nactions = self.nactions)


    def is_state(self, s):
        return s in self.rstates

    def state_index(self, s):
        return self.rstates[s]

    def coords(self, s):
        return (s / self.ncols, s % self.ncols)

    def dcoords(self, a):
        # turn action index into an action A st. CS + A = NS
        actions = [np.array((-1,0)),np.array((-1,-1)),
                   np.array((0,-1)),np.array((1,-1)),
                   np.array((1,0)), np.array((1,1)),
                   np.array((0,1)), np.array((-1,1))]

        return actions[a]

    def neighbors(self, state):
        result = []
        for i in self.actions:
            result.append(self.get_next_state(i,state))
        return result

    def distance(self, start, end):
        """ Use euclidean distance here? """
        return la.norm(np.array(self.states[start]) - np.array(self.states[end]))

    def shortest_path(self, start, end):
        from astar import astar
        path = astar(self.neighbors, start, end, lambda x,y : 0.0, self.distance)
        actions = [self.get_action(path[i],path[i+1]) for i in range(len(path) - 1)]
        return path, actions

    def get_action(self, i, j):
        """ Get action that connects states i,j if available. """
        for a in self.actions:
            if j == self.get_next_state(a,i):
                return a

    def get_next_state(self, a, i):
        """ Fast next state computation for deterministic models. """

        cs = self.states[i]
        ac = self.dcoords(a)
        ns = cs + ac
        ns = tuple(ns)

        if self.is_state(ns):
            return self.state_index(ns)
        else:
            return i

    def get_reward(self, i):
        if i in self.endstates:
            return 1.0
        else:
            return 0.0


class Gridworld8( MDP ):
    """
    This is a rather unfancy gridworld that extends the basic discrete
    MDP framework to parameterize the size of the gridworld. Subclass
    this for more advanced gridworlds with "walls" etc.

    A good way to add obstacles is to define a set of indices with a
    good descriptive name, and then deal with those special cases in
    the initialization functions for the transition and reward
    models. See for example the way the boundaries are delt with.
    """

    def __init__(self, nrows = 5, ncols = 5, walls=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)], endstates = [0]):
        self.nrows = nrows
        self.ncols = ncols

        self.walls = walls
        grid = [self.coords(i) for i in range(self.nrows * self.ncols)]
        grid = [s for s in grid if not s in self.walls]
        self.states = dict([(i,s) for (i,s) in enumerate(grid)])
        self.rstates = dict([(s,i) for (i,s) in enumerate(grid)]) # reverse lookup by grid coords

        self.nstates = len(self.states)
        self.nactions = 8
        self.endstates = endstates

        MDP.__init__(self, nstates = self.nstates, nactions = self.nactions)

    def is_state(self, s):
        return s in self.rstates

    def state_index(self, s):
        return self.rstates[s]

    def coords(self, s):
        return (s / self.ncols, s % self.ncols)

    def dcoords(self, a):
        # turn action index into an action A st. CS + A = NS
        actions = [np.array((-1,0)),np.array((-1,-1)),
                   np.array((0,-1)),np.array((1,-1)),
                   np.array((1,0)), np.array((1,1)),
                   np.array((0,1)), np.array((-1,1))]

        return actions[a]

    # initialize_* methods create a complete model -- inefficient
    def initialize_rewards(self, a, i, j):

        if j in self.endstates:
            return 1.0
        else:
            return 0.0

    def initialize_model(self, a, i, j):
        cs = self.states[i]
        ac = self.dcoords(a)
        ns = cs + ac
        ns = tuple(ns)

        if self.is_state(ns):
            if (ns == self.states[j]):
                return 1.0
            else:
                return 0.0
        else:
            if (i == j):
                return 1.0
            else:
                return 0.0

if __name__ == '__main__':

    gw = Gridworld8(walls = [(0,2),(1,2),(3,2),(4,2)])
    gws = SparseGridworld8(nrows = 32, ncols = 64)
    #bad = Gridworld8(nrows = 32, ncols=64) # will blowup memory
    t = gws.trace(1000)
