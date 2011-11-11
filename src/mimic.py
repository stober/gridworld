#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: MIMIC.PY
Date: Thursday, April 28 2011

Description: The mimic class creates a pseudo-mdp based on a trace,
then provides the same inferface. The difference is that nothing is
simulated, the trace is just continually replayed. The idea is that
you can eliminate the randomness from computations and then compare
results for specific differences.
"""

import numpy as np

class Mimic(object):

    def __init__(self, trace, limit):
        self.trace = trace
        self.step = 0
        self.pstep = 0
        self.nfeatures = 4 # hack for CW env
        self.limit  = limit

    def random_policy(self):
        action = self.trace[self.step][1]

    def vphi(self, state):
        features = np.zeros(self.nfeatures)
        features[state] = 1.0
        return features

    def move(self,action):
        answer = self.trace[self.step]
        self.step += 1
        if self.step >= len(self.trace):
            self.step = 0
        return answer

    def simulate(self):
        previous, action, reward, next = self.move(0)
        return previous, reward, next

    def terminal(self, state):
        # hack for CW environment
        self.pstep += 1
        if self.pstep > self.limit:
            return True
        else:
            return False

    def reset(self):
        self.current = self.trace[self.step][0]
        self.pstep = 0

