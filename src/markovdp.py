#! /usr/bin/env python
"""
Author: Jeremy M. Stober
Program: MDP.PY
Date: Monday, January 11 2010
Description: Basic MDP framework for Markov Decision Problems.
"""

import os, sys, getopt, pdb, string

import functools
import random as pr
import numpy as np
import numpy.random as npr

class MDP( object ):

    def __init__(self, nstates = 32, nactions = 4):
        self.nstates = nstates
        self.sindices = range(self.nstates)
        self.nactions = nactions
        self.actions = range(self.nactions)
        self.gamma = 0.9

        # possible start states for reset problems
        if not hasattr(self, 'startindices'):
            self.startindices = range(self.nstates)

        # possible end states that initiate resets
        if not hasattr(self, 'endstates'):
            self.endstates = []

        # model tensor is initialized from a function that needs to be overwritten in subclasses

        indices = [(a,i,j) for a in self.actions for i in self.sindices for j in self.sindices]
        shape = (self.nactions,self.nstates,self.nstates)

        self.model = np.array([self.initialize_model(*idx) for idx in indices]).reshape(*shape)
        self.rewards = np.array([self.initialize_rewards(*idx) for idx in indices]).reshape(*shape)

        self.reset()

        # the tabular number of features (generic)
        self.tab_nfeatures = self.nstates * self.nactions + 1

        # the polynomial number of features (generic)
        self.maxexp = 9
        self.poly_nfeatures = self.nactions * self.maxexp + 1

        # the indicator representation of phi (generic)
        self.ind_nfeatures = self.nstates + self.nactions

        # the hash method of representation (generic)
        self.hash_nfeatures = 16
        self.hash_prime = 191 # should be bigger than self.nstates * self.nactions?
        self.a = 36
        self.b = 105

        # Both the transition model and rewards are tensors. For a
        # particular action, the model tensor resolves to a stochastic
        # matrix. The reward tensor should be sparse (unlike the value
        # function tensor).

        # In particular, if generating random problem instances, care
        # must be taken to make sure that the distribution over
        # transition models and rewards "makes sense."

        # The default form of the random model will be that of a
        # "gridworld."  Actions are "left","up","right","down" and
        # result in non-zero transition probabilities for the
        # appropriate neighboring states in particular cases.

    def initial_policy(self):
        return np.zeros(self.tab_nfeatures)

    def initialize_rewards(self, a, i, j):
        if j == 31:
            return 1.0
        else:
            return 0.0

    def is_markov_process(self):
        return len(self.nactions) == 1

    def initialize_model(self, a, i, j):
        """
        This function fills in the model tensor for each pair of
        states and action. This needs to be overwritten when
        subclassing MDP.
        """

        if a == 0:
            # left

            # left edge
            if i in (0,8,16,24):
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

            # top edge
            if 0 <= i <= 7:
                if i == j:
                    return 1.0
                else:
                    return 0.0

            elif j == i - 8:
                return 1.0

            else:
                return 0.0

        elif a == 2:
            # right

            # right edge
            if i in (7,15,23,31):
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

            # bottom edge
            if 24 <= i <= 31:
                if i == j:
                    return 1.0
                else:
                    return 0.0

            elif j == i + 8:
                return 1.0

            else:
                return 0.0

    def state(self):
        return self.current

    def observe(self, s):
        pass

    def move(self, a, obs=False):
        self.previous = self.current
        transition_distribution = self.model[a,self.current]

        assert np.sum(transition_distribution) == 1.0, "a: %d, s: %d, p: %f" % (a, self.current, np.sum(transition_distribution))

        choice = npr.multinomial(1,transition_distribution)
        nonzero = np.nonzero(choice)[0]
        assert len(nonzero) == 1
        self.current = nonzero[0]

        self.reward =  self.rewards[a,self.previous,self.current]

        if obs:
            return (self.observe(self.previous), a, self.reward, self.observe(self.current))
        else:
            return (self.previous, a, self.reward, self.current)

    def linear_policy(self, w,s):
        # note that phi should be overridden (or learned in some way)
        return np.argmax([np.dot(w,self.phi(s,a)) for a in range(self.nactions)])

    def evaluate_policy(self, w):
        policy = functools.partial(self.linear_policy, w)
        traces = []
        for s in self.startindices:
            traces.append(self.single_trace(s,policy))

        # need to evaluate traces
        return traces

    def terminal(self):
        # override in environments that have or need a terminal state
        pass

    def biased_choice(self, choices, pvalues):
        markers = []
        acc = 0
        for val in pvalues:
            acc += val
            markers.append(acc)

        r = pr.random()
        print markers, r
        return np.searchsorted(markers, r)

    def softmax(self, values):
        e = np.exp(values)
        d = np.sum(e)
        p = e / d
        print p
        return self.biased_choice(self.actions, p)

    def soft_linear_policy(self, w, s):
        qa = []
        for a in range(self.nactions):
            f = self.phi(s, a)
            qa.append(np.dot(w, f))
        return self.softmax(qa)

    def callback(self, *args, **kwargs):
        pass

    def ind_phi(self, s, a):
        features = np.zeros(self.ind_nfeatures)
        features[s] = 1.0
        features[self.nstates + a] = 1.0
        return features

    def tab_phi(self, s, a):
        # action features - default is a tabular representation
        features = np.zeros(self.tab_nfeatures)
        features[s + (a * self.nstates)] = 1.0
        features[-1] = 1.0
        return features

    def poly_phi(self, s, a):
        features = np.zeros(self.poly_nfeatures)

        for i in range(self.maxexp):
            features[a * self.maxexp + i] = s ** i

        features[-1] = 1.0
        return features

    def hash_phi(self, s, a):
        features = np.zeros(self.hash_nfeatures)
        x = (s + 1) * (a + 1)
        f = (self.a * x + self.b) % self.hash_prime
        g = f % self.hash_nfeatures
        features[g] = 1.0
        return features

    def nfeatures(self):
        return self.tab_nfeatures

    def phi(self, s, a):
        return self.tab_phi(s,a)

    def vphi(self, s):
        # just a tabular version of the phi function at the moment
        features = np.zeros(self.nstates + 1)
        features[s] = 1.0
        features[-1] = 1.0
        return features

    def generate_value_function(self, w):
        qfunc = {}
        vfunc = {}
        for s in range(self.nstates):
            maxq = -1e6
            for a in range(self.nactions):
                value = np.dot(w, self.phi(s, a))
                qfunc[(s,a)] = value
                if value > maxq:
                    maxq = value
            vfunc[s] = value
        return qfunc,vfunc

    def generate_policy(self, w):
        policy = {}
        for s in range(self.nstates):
            a = self.linear_policy(w,s)
            policy[s] = a
        return policy

    def generate_all_policies(self, w, threshold = 1e-6):
        """
        Enumerate all (equivalent) policies given the current weights.
        """

        all_policies = {}
        for s in range(self.nstates):
            values = np.array([np.dot(w, self.phi(s,a)) for a in range(self.nactions)])

            # this is basically like selecting all indices within a tolerance of the max value
            actions = np.nonzero(np.max(values) - values < threshold)
            all_policies[s] = actions[0]

        return all_policies

    def random_policy(self, *args):
        # random policy
        return pr.choice(self.actions)

    def generate_wrandom_policy(self, w):
        # accumulate the weights for all the actions
        bins = np.cumsum(w)

        def policy(*args):
            value = pr.random()
            ind = np.searchsorted(bins,value)
            return self.actions[ind]

        return policy

    def reset(self):
        self.current = pr.choice(self.startindices)
        self.previous = -1
        self.action = -1
        self.reward = 0

    def identity(self, arg):
        return arg

    def single_episode(self, policy = None, obs = False):
        self.reset()
        if policy is None: policy = self.random_policy

        if obs:
            observe = self.observe
        else:
            observe = self.identity

        trace = []
        # initialize the actions
        next_action = policy(observe(self.current))
        while not self.current in self.endstates:
            pstate, paction, reward, state = self.move(next_action, obs=obs)
            next_action = policy(observe(self.current)) # next state
            trace.append((pstate, paction, reward, state, next_action))

        return trace

    def trace(self, tlen = 1000, policy=None, obs = False):
        # generate a trace using whatever policy is currently implemented

        if policy is None: policy = self.random_policy

        if obs:
            observe = self.observe
        else:
            observe = self.identity

        trace = []
        next_action = policy(observe(self.current))
        for i in range(tlen):

            pstate, paction, reward, state = self.move(next_action, obs=obs)
            trace.append((pstate, paction, reward, state, next_action))

            if self.current in self.endstates:
                self.reset()
                next_action = policy(observe(self.current))

        return trace

class FastMDP(MDP):
    """
    There is a useful subcase of MDPs with large state spaces and
    deterministic actions which we may wish to model. In these cases
    it does not make sense to compute the entire model in advance.
    """

    def __init__(self, nstates = 32, nactions = 4):
        self.nstates = nstates
        self.sindices = range(self.nstates)
        self.nactions = nactions
        self.actions = range(self.nactions)

        # the tabular number of features (generic)
        self.tab_nfeatures = self.nstates * self.nactions + 1

        if not hasattr(self, 'startindices'):
            self.startindices = range(self.nstates)

        # possible end states that initiate resets
        if not hasattr(self, 'endstates'):
            self.endstates = []

        self.model = np.zeros((nactions,nstates),dtype='int')
        for a in self.actions:
            for i in self.sindices:
                self.model[a,i] = self.get_next_state(a,i)

        self.rewards = np.zeros(nstates)
        for i in self.sindices:
            self.rewards[i] = self.get_reward(i)

        self.reset()

    def move(self, a, obs=False):
        self.previous = self.current
        
        self.current = self.model[a,self.previous]
        self.reward = self.rewards[self.current]

        if obs:
            return (self.observe(self.previous), a, self.reward, self.observe(self.current))
        else:
            return (self.previous, a, self.reward, self.current)


class MP( MDP ):
    """
    Sort of strange for a Markov Process to inherit from a MDP, but
    since a MP is an MDP with one action, this method seems to
    work. The idea is just to implement simulate instead of move. and
    to limit the number of possible actions to 1.
    """

    def __init__(self, nstates = 32):
        MDP.__init__(self, nstates = nstates, nactions = 1)


    def simulate(self):
        """
        Simulate a single step in the Markov process.
        """

        previous, action, reward, next = self.move(0)
        return previous, reward, next # action == 0
