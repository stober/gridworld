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
from markovdp import MDP,FastMDP,SparseMDP, Features, AliasFeatures
from utils import sp_create, sp_create_data

def coords(nrows,ncols,s):
    return (s / ncols, s % ncols)

def wall_pattern(nrows,ncols,endstate=0,pattern="comb"):
    """Generate a specific wall pattern for a particular gridworld."""

    goal = coords(nrows,ncols,endstate)
    walls = []

    if goal[1] % 2 == 0:
        wmod = 1            
    else:
        wmod = 0

    for i in range(ncols):
        for j in range(nrows):
            if i % 2 == wmod and j != nrows - 1:
                walls.append((i,j))

    return walls

class ObserverFeatures( Features ):

    def nfeatures(self):
        return len(self.observations[0]) * self.nactions + 1

    def phi(self, s, a, sparse=False, format="csr"):
        """ sparse not implemented """
        features = np.zeros(self.nfeatures())
        obs = self.observe(s)
        features[a * len(obs) : a*len(obs) + len(obs)] = obs
        features[-1] = 1.0
        return features

class DiscreteObserverFeatures( Features ):

    def __init__(self, nbuckets = 100, nobs = 10):
        self.obs = self.observations
        self.nbuckets = nbuckets
        self.nobs = nobs # only tile first 5 buckets

        self.bins = []
        for i in range(nobs):
            minval = np.min(self.obs[:,i])
            maxval = np.max(self.obs[:,i])
            dist = maxval - minval
            interval = float(dist) / float(nbuckets)
            bins = np.arange(minval + interval, maxval, interval)
            self.bins.append(bins)

    def tile(self, s, a):
        obs = self.obs[s]
        idx = []

        for (i,bins) in enumerate(self.bins):
            binx = i * self.nbuckets
            offset = a  *  (self.nbuckets * self.nobs) +  binx
            idx.append(np.searchsorted(bins, obs[i]) + offset)

        return idx

    def nfeatures(self):
        return self.nbuckets * self.nobs * self.nactions + 1

    def phi(self, s, a, sparse=False, format="csr"):
        obs = self.observe(s) # 100 pca weights

        if not sparse:
            features = np.zeros(self.nfeatures())
            features[-1] = 1.0
            indx = self.tile(s,a)
            features[indx] = 1.0
            return features
        else:
            cols = np.array([0] * self.nobs + [0])
            rows = np.array(self.tile(s,a) + [self.nfeatures()  - 1])
            data = np.array([1.0] * self.nobs + [1.0])
            sparse_features = sp_create_data(data, rows, cols, self.nfeatures(),1,format)
            return sparse_features


class RBFFeatures( Features ):

    def __init__(self, nrows, ncols, nactions, nrbf):
        self.nrows = nrows
        self.ncols = ncols
        self.nrbf = nrbf
        self.feature_cnt = nrbf * nactions + 1
        self.rbf_loc = pr.sample([(i,j) for i in range(self.nrows) for j in range(self.ncols)], nrbf)

    def nfeatures(self):
        return self.feature_cnt

    def coords(self, s):
        """
        Location of state in gridworld coordinates.
        """
        return (s / self.ncols, s % self.ncols)

    def rfunc(self, s):
        """
        Compute the responses. Override for different resposne functions.
        """
        c = np.array(self.coords(s))
        return [ np.exp(-.001 * la.norm(c - i, ord=np.inf) ** 2) for i in self.rbf_loc ]

    def phi(self, s, a, sparse=False, format="csr"):
        if sparse:
            cols = np.array([0] * (self.nrbf + 1))
            rows = np.array([a * self.nrbf + i for i in range(self.nrbf)] + [self.feature_cnt - 1])
            data = np.array(self.rfunc(s) + [1.0])
            sparse_features = sp_create_data(data,rows,cols,self.feature_cnt,1,format)
            return sparse_features
        else:
            features = np.zeros(self.feature_cnt)
            for i,f in enumerate(self.rfunc(s)):
                features[a * self.nrbf + i] = f
            features[-1] = 1.0
            return features

class SparseGridworld8( SparseMDP ):

    def __init__(self, nrows = 5, ncols = 5, actions = None, walls=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)], endstates = [0]):
        self.nrows = nrows
        self.ncols = ncols

        self.walls = walls
        grid = [self.coords(i) for i in range(self.nrows * self.ncols)]
        grid = [s for s in grid if not s in self.walls]
        self.states = dict([(i,s) for (i,s) in enumerate(grid)])
        self.rstates = dict([(s,i) for (i,s) in enumerate(grid)]) # reverse lookup by grid coords

        if actions is None:
            actions = range(8)

        self.allowed_actions = actions
        self.nstates = len(self.states)
        self.nactions = len(actions)
        self.endstates = endstates

        SparseMDP.__init__(self, nstates = self.nstates, nactions = self.nactions)


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

        return actions[self.allowed_actions[a]]

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

class SparseRBFGridworld8( SparseGridworld8, RBFFeatures ):

    def __init__(self, nrows = 5, ncols = 5, walls=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)], endstates = [0], nrbf = 15):
        SparseGridworld8.__init__(self, nrows=nrows, ncols=ncols, walls=walls, endstates=endstates)
        RBFFeatures.__init__(self, self.nrows, self.ncols, self.nactions, nrbf)

class SparseAliasGridworld8( SparseGridworld8, AliasFeatures ):
    def __init__(self, nrows = 5, ncols = 5, walls=[(1,1),(1,2),(1,3),(2,1),(2,2),(2,3),(3,1),(3,2),(3,3)], endstates = [0], aliases={}):
        SparseGridworld8.__init__(self, nrows=nrows, ncols=ncols, walls=walls, endstates=endstates)
        AliasFeatures.__init__(self, self.nstates, self.nactions, aliases)

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


class ObserverGridworld(SparseGridworld8, DiscreteObserverFeatures):
    """
    A gridworld overlayed with some pregenerated observations. 
    """

    def __init__(self, observations, coordinates, walls = None, endstates = None):
        self.observations = np.load(observations) # KxL matrix of observations
        self.coordinates = np.load(coordinates) # Nx2 matrix of coordinates that can be mapped to a gridworld

        # determine the indices for everything in self.coordinates
        xcoords = set(self.coordinates[:,0])
        ycoords = set(self.coordinates[:,1])

        # need to make sure ordering of coordinates is correct?
        nrows = len(xcoords)
        ncols = len(ycoords)

        # For gw state indices (and dynamics) to match to provided data, the
        # order of the coordinates needs to also match. That means the data is
        # in row major order. The following sorting insures that this is true.

        ind = np.lexsort((self.coordinates[:,1], self.coordinates[:,0]))
        self.observations = self.observations[ind]
        self.coordinates = self.coordinates[ind]

        if walls is None:
            walls = []

        if endstates is None:
            endstates = [0]

        super(ObserverGridworld, self).__init__(nrows=nrows, ncols = ncols, endstates = endstates, walls = walls)
        DiscreteObserverFeatures.__init__(self)

    def observe(self,s):
        return self.observations[s]

if __name__ == '__main__':

    # gw = Gridworld8(walls = [(0,2),(1,2),(3,2),(4,2)])
    # gws = SparseGridworld8(nrows = 32, ncols = 64)
    # bad = Gridworld8(nrows = 32, ncols=64) # will blowup memory
    # t = gws.trace(1000)
    # gw = SparseRBFGridworld8(nrows = 32, ncols = 32)
    # t = gw.trace(1000)

    obs = np.load("/Users/stober/wrk/lspi/bin/projections.npy")
    for (i,p) in enumerate(obs):
        for (j,q) in enumerate(obs):
            if i != j and np.allclose(p,q):
                print i,j," match!"

    # ogw = ObserverGridworld("/Users/stober/wrk/lspi/bin/observations4.npy", "/Users/stober/wrk/lspi/bin/states.npy")
    # all_phi = [ogw.phi(s,a) for s in ogw.states for a in ogw.actions]
    # for (i,p) in enumerate(all_phi):
    #     for (j,q) in enumerate(all_phi):
    #         if i != j and np.allclose(p,q):
    #             print i,j," match!"


