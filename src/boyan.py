#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: BOYAN_EXAMPLE.PY
Date: Tuesday, January 12 2010
Description: Simple MDP from Boyan 2002.
"""

import os, sys, getopt, pdb, string

import numpy as np
import numpy.random as npr
import random as pr
import numpy.linalg as la
import matplotlib as mpl

from markovdp import MP

class Boyan(MP):

    def __init__(self):
        self.nfeatures = 4
        self.statefeatures = np.array([[0,0,0,1],[0,0,0.25,0.75], [0,0,0.5,0.5],[0,0,0.75,0.25],
                                       [0,0,1,0],[0,0.25,0.75,0], [0,0.5,0.5,0],[0,0.75,0.25,0],
                                       [0,1,0,0],[0.25,0.75,0,0], [0.5,0.5,0,0],[0.75,0.25,0,0],
                                       [1,0,0,0]], dtype=float)
        self.actionfeatures = np.array([[0,1],[1,0]], dtype = float)
        self.endstates = [0]
        MP.__init__(self, nstates = 13)

    def terminal(self, state):
        return state == 0

    def initialize_model(self, a, i, j):

        if i == 1 and j == 0:
            return 1.0
        elif i - 1 == j:
            return 0.5
        elif i - 2 == j:
            return 0.5
        elif i == 0 and j == 0:
            return 1.0
        else:
            return 0.0

    def initialize_rewards(self, a, i, j):

        if i == 1 and j == 0:
            return -2.0
        elif i - 1 == j:
            return -3.0
        elif i - 2 == j:
            return -3.0
        else:
            return 0.0

    def vphi(self, state):
        if self.terminal(state):
            return np.zeros(4)
        else:
            return self.statefeatures[state]
