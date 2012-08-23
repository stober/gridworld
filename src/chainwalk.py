#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: CHAINWALK.PY
Date: Monday, January 11 2010
Description: Chainwalk from LSPI paper (2003).
"""

import os, sys, getopt, pdb, string
import random as pr
import numpy as np
import scipy as sp
import scipy.io as sio

from markovdp import MDP

class Chainwalk( MDP ):

    def __init__(self):

        # actions:
        # 0 == L
        # 1 == R

        self.gamma = 0.9
        self.feature_cnt = 9
        MDP.__init__(self, nstates = 4, nactions = 2)

    def reset(self):
        self.step = 0
        return MDP.reset(self)

    def initialize_rewards(self, a, i ,j):
        if i in (1,2):
            return 1.0
        else:
            return 0.0

    def terminal(self,state):
        # hack to get episodic samples in LSTD without a terminal state
        self.step += 1
        if self.step > 500:
            return True
        else:
            return False


    def initialize_model(self, a, i, j):

        L = [[0.9,0.1,0.0,0.0],
             [0.9,0.0,0.1,0.0],
             [0.0,0.9,0.0,0.1],
             [0.0,0.0,0.9,0.1]]

        R = [[0.1,0.9,0.0,0.0],
             [0.1,0.0,0.9,0.0],
             [0.0,0.1,0.0,0.9],
             [0.0,0.0,0.1,0.9]]

        if a == 0:
            return L[i][j]
        else:
            return R[i][j]

    def vphi(self, state):
        features = np.zeros(self.nstates)
        features[state] = 1.0
        return features

class ChainwalkPoly( Chainwalk ):

    def initial_policy(self):
        return np.zeros(6)

    def phi(self, s, a):
        s = s + 1.0
        s = 10.0 * s / 4.0

        if a == 0:
            return np.array([1.0,s,s**2,0.0,0.0,0.0])
        else:
            return np.array([0.0,0.0,0.0,1.0,s,s**2])

