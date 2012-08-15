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
import scipy.sparse as sp

class Features( object ):
    """
    Mixin for specific feature functions. Default is tabular feature
    representation.
    """

    def __init__(self, nstates, nactions):
        self.nstates = nstates
        self.nactions = nactions
        self.feature_cnt = nstates * nactions + 1
        self.features = np.zeros(self.feature_cnt)
        self.sparse_features = sp.dok_matrix((self.feature_cnt,1))

    def nfeatures(self):
        return self.feature_cnt

    def phi(self, s, a, sparse=False):
        if sparse:
            self.sparse_features.clear()
            self.sparse_features[s + (a * self.nstates),0] = 1.0
            self.sparse_features[-1,0] = 1.0
            return self.sparse_features
        else:
            self.features[:] = 0
            self.features[s + (a * self.nstates)] = 1.0
            self.features[-1] = 1.0
            return self.features

    def vphi(self, s):
        features = np.zeros(self.nstates + 1)
        features[s] = 1.0
        features[-1] = 1.0
        return features

class PolyFeatures( Features ):
    """
    Polynomial feature representation.
    """

    def __init__(self, nstates, nactions, maxexp = 9 ):
        self.nstates = nstates
        self.nactions = nactions
        self.maxexp = maxexp
        self.feature_cnt = nstates * maxexp + 1
        self.features = np.zeros(self.feature_cnt)

    def phi(self, s, a):
        self.features[:] = 0
        for i in range(self.maxexp):
            self.features[a * self.maxexp + i] = s ** i

        self.features[-1] = 1.0
        return features

    def vphi(self, s):
        features = np.zeros(self.maxexp + 1)
        for i in range(self.maxexp):
            features[i] = s ** i
        features[-1] = 1.0
        return features

class IndicatorFeatures( Features ):
    """
    Indicate the state and the action.
    """
    def __init__(self, nstates, nactions):
        self.nstates = nstates
        self.nactions = nactions
        self.feature_cnt = nstates + nactions + 1
        self.features = np.zeros(self.feature_cnt)

    def phi(self, s, a):
        self.features[:] = 0
        self.features[s] = 1.0
        self.features[nstates + a] = 1.0
        self.features[-1] = 1.0
        return self.features

    def vphi(self, s):
        features = np.zeros(self.nstates + 1)
        features[s] = 1.0
        features[-1] = 1.0
        return features

class HashFeatures( Features ):
    """
    Hash features.
    """
    def __init__(self, nstates, nactions, nfeatures = 15 ):
        self.nstates = nstates
        self.nactions = nactions
        self.feature_cnt = nfeatures
        self.prime = 191
        self.a = 36
        self.b = 105
        self.features = np.zeros(self.feature_cnt)

    def phi(self, s, a):
        self.features[:] = 0
        x = (s + 1) * (a + 1)
        f = (self.a * x + self.b) % self.prime
        g = f % self.feature_cnt
        self.features[g] = 1.0
        return self.features


    def vphi(self, s):
        features = np.zeros(self.feature_cnt)
        x = (s + 1)
        f = (self.s * x + self.b) % self.prime
        g = f % self.feature_cnt
        features[g] = 1.0
        return features
    
class MDP( Features ):

    def __init__(self, nstates = 32, nactions = 4):
        self.nstates = nstates
        self.sindices = range(self.nstates)
        self.nactions = nactions
        self.actions = range(self.nactions)
        self.vcnts = np.zeros(self.nstates)
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

        Features.__init__(self, self.nstates, self.nactions) # feature mixin supplies phi and related methods

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

    def value_iteration(self):
        """
        The optimal value function (and optimal policy) can be computed from the model directly using dynamic programming.
        """
        # compute values over states
        values = np.zeros(self.nstates)
        rtol = 1e-4

        # compute the value function
        condition = True
        i = 0
        while condition:
            delta = 0
            for s in self.sindices:
                v = values[s]
                sums = np.zeros(self.nactions)
                for a in self.actions:
                    for t in self.sindices:
                        sums[a] += self.model[a,s,t] * (self.rewards[a,s,t] + self.gamma * values[t])
                
                values[s] = np.max(sums)
                delta = max(delta, abs(v - values[s]))
            print i,delta
            i += 1
            if delta < rtol:
                break

        # compute the optimal policy
        policy = np.zeros(self.nstates, dtype=int) # record the best action for each state
        for s in self.sindices:
            sums = np.zeros(self.nactions)
            for a in self.actions:
                for t in self.sindices:
                    sums[a] += self.model[a,s,t] * (self.rewards[a,s,t] + self.gamma * values[t])
            policy[s] = np.argmax(sums)

        return values,policy

    def reset_counts(self):
        self.vcnts = np.zeros(self.nstates)

    def set_state(self, state):
        self.current = state
        self.vcnts[self.current] += 1

    def initial_policy(self):
        return np.zeros(self.nfeatures())

    def initialize_reward(self, a, i, j):
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

    def failure(self):
        if self.current in self.endstates:
            return True
        else:
            return False

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

    def reset(self, method = "random"):
        if method == "random":
            self.current = pr.choice(self.startindices)
        else: # choose least visited state
            self.current = np.argmin(self.vcnts)

        self.previous = -1
        self.action = -1
        self.reward = 0
        self.vcnts[self.current] += 1

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
            next_action = policy(observe(self.current))
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
        self.vcnts = np.zeros(self.nstates)
        self.gamma = 0.9

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

        Features.__init__(self, self.nstates, self.nactions) # feature mixin supplies phi and related methods

    @staticmethod
    def load(filename):
        import re
        fp = open(filename)

        nactions = -1
        nstates = -1
        endstates = []
        model = None
        rewards = None

        state = 'nostate'

        for line in fp.readlines():
            if state == 'nostate':
                if re.match(r'nactions', line):
                    _,nactions = line.split()
                    nactions = int(nactions)
                elif re.match(r'nstates', line):
                    _,nstates = line.split()
                    nstates = int(nstates)
                elif re.match(r'endstates', line):
                    endstates = [int(i) for i in line.split()[1:]]
                elif re.match(r'rewards', line):
                    state = 'rewards'
                    rewards = np.zeros(nstates)
                elif re.match(r'model', line):
                    state = 'model'
                    model = np.zeros((nactions,nstates))
            elif state == 'model':
                s,a,n = line.split()
                model[int(a),int(s)] = int(n)
            elif state == 'rewards':
                s,r = line.split()
                rewards[int(s)] = r

        instance = FastMDP.__init__(nstates = nstates, nactions = nactions)
        instance.endstates = endstates
        instance.model = model
        instance.rewards = rewards

        return instance

    def save(self, filename):
        """
        Save MDP in custom format for easy inspection.
        """

        fp = open(filename, 'w')
        fp.write('FastMDP\n\n')
        fp.write("nactions {0}\n".format(self.nactions))
        fp.write("nstates {0}\n\n".format(self.nstates))

        fp.write("endstates ")
        for i in self.endstates:
            fp.write("{0} ".format(i))

        fp.write("\n\nrewards\n")
        for i in self.sindices:
            fp.write("{0} {1}\n".format(i,self.rewards[i]))

        fp.write("\nmodel\n")
        for i in self.sindices:
            for a in self.actions:
                fp.write("{0} {1} {2}\n".format(i, a, self.model[a,i]))


    def move(self, a, obs=False):
        self.previous = self.current

        self.current = self.model[a,self.previous]
        self.reward = self.rewards[self.current]

        self.vcnts[self.current] += 1

        if obs:
            return (self.observe(self.previous), a, self.reward, self.observe(self.current))
        else:
            return (self.previous, a, self.reward, self.current)


class SparseMDP( MDP ):
    """
    Reward and model transition distributions are often sparse
    (esp. for big problems) and so it makes sense to use sparse
    matrices instead of dense matrices for these.
    """

    def __init__(self, nstates = 32, nactions = 4):
        self.nstates = nstates
        self.sindices = range(self.nstates)
        self.nactions = nactions
        self.actions = range(self.nactions)
        self.vcnts = np.zeros(self.nstates)
        self.gamma = 0.9
        
        # possible start states for reset problems
        if not hasattr(self, 'startindices'):
            self.startindices = range(self.nstates)

        # possible end states that initiate resets
        if not hasattr(self, 'endstates'):
            self.endstates = []

        self.model = {}
        for a in self.actions:
            for i in self.sindices:
                self.model[a,i] = self.initialize_model(a,i)

        self.rewards = self.initialize_rewards()

        self.reset()

        Features.__init__(self, self.nstates, self.nactions) # feature mixin supplies phi and related methods

    def move(self, a, obs=False):
        self.previous = self.current
        distribution = self.model[a,self.current] # list of (state, prob) pairs
        states = [x[0] for x in distribution]
        probs = [x[1] for x in distribution]

        assert np.sum(probs) == 1.0, "a: %d, s: %d, p: %f" % (a, self.current, np.sum(transition_distribution))
        choice = npr.multinomial(1,probs)
        nonzero = np.nonzero(choice)[0]
        assert len(nonzero) == 1

        self.current = states[nonzero[0]]

        self.reward =  self.rewards[self.current]

        if obs:
            return (self.observe(self.previous), a, self.reward, self.observe(self.current))
        else:
            return (self.previous, a, self.reward, self.current)

    def value_iteration(self):
        """
        The optimal value function (and optimal policy) can be computed from the model directly using dynamic programming.
        """
        # compute values over states
        values = np.zeros(self.nstates)
        rtol = 1e-4

        # compute the value function
        condition = True
        i = 0
        while condition:
            delta = 0
            for s in self.sindices:
                v = values[s]
                sums = np.zeros(self.nactions)
                for a in self.actions:
                    dist = self.model[a,s]
                    for (t,p) in dist:
                        sums[a] += p * (self.rewards[t] + self.gamma * values[t])
                values[s] = np.max(sums)
                delta = max(delta, abs(v - values[s]))
            print i,delta
            i += 1
            if delta < rtol:
                break

        # compute the optimal policy
        policy = np.zeros(self.nstates, dtype=int) # record the best action for each state
        for s in self.sindices:
            sums = np.zeros(self.nactions)
            for a in self.actions:
                dist = self.model[a,s]
                for (t,p) in dist:
                    sums[a] += p * (self.rewards[t] + self.gamma * values[t])
            policy[s] = np.argmax(sums)

        return values,policy


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
