################################################################################
#                                                                              #
# Statistical Machine Learning & AI                                            #
#                                                                              #
# Problem:  Contextual bandits                                                 #
#                                                                              #
# Author:   Anubhav Singh                                                      #
#                                                                              #
# References:                                                                  #
#   1. Linear UCB: https://arxiv.org/pdf/1003.0146.pdf                         #
#                                                                              #
################################################################################

from mab import MAB
from multiprocessing import Pool
from multiprocessing import cpu_count
from functools import partial
import numpy as np

def argmax_rand(a):
    """
    Arguments
    =========
    a : numpy float array, 1-D of length 'narms'

    Returns:
    ========
    index : int
        Randomly pick an index from the set of
        indices corresponding to a's max value.
    """
    return np.random.choice(np.flatnonzero(a == a.max()))

class LinUCB(MAB):
    """
    Contextual multi-armed bandit (LinUCB)

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    alpha : float
        positive real explore-exploit parameter
    """
    def __init__(self, narms, ndims, alpha):
        self.narms          = narms
        self.ndims          = ndims
        self.alpha          = alpha
        self.arm_at_t       = {}
        # A[arm] represents (X.X_transpose + lambda*I) from linear reg.
        self.A              = {}
        # B[arm] represents rewards vector (X_transponse.Y) from linear reg.
        self.B              = {}
        self.num_times_arm  = {}

        for i in range(self.narms):
            # Initialize A as Identity Matrix
            self.A[i+1] = np.identity(ndims)
            # Initialize B as Zero Vector
            self.B[i+1] = np.zeros((ndims,1))

    def play(self, tround, context):
        """
        Returns a selected arm

        Arguments
        =========
        tround : int
            the positive integer arm id for this round
        context : 1D float array, shape (self.ndims * self.narms),
            optional context given to the arms

        Returns
        =======
        arm : int
            the positive integer arm id for this round
        """

        # Reset if the play call is for same round number as previous one
        if tround in self.arm_at_t.keys():
            tround_repeat = True
            self.num_times_arm[self.arm_at_t[tround]]-=1
        else:
            tround_repeat = False

        ## Comment the below lines for parallel execution
        p   = [None]*self.narms #captures expected reward for each arm
        for i in range(self.narms):
            p[i]    =   util_calc_reward(context[10*i:10*i+10], self.alpha,
                                         self.A[i+1], self.B[i+1])
        p = np.array(p)

        ## Parallel code (Slower!!)
        ## Uncomment for parallel execution
        # data_p  = [(context[10*i:10*i+10], self.alpha, self.A[i+1],
        #                 self.B[i+1])  for i in range(self.narms)]
        # p       = np.array(process_pool.starmap(util_calc_reward, data_p))

        # Random tie-break against all index corresponding to max value of Q
        self.arm_at_t[tround] = argmax_rand(p)

        # Store the selected arm
        self.num_times_arm[self.arm_at_t[tround]] = \
                self.num_times_arm.setdefault(self.arm_at_t[tround],0)+1

        return self.arm_at_t[tround]+1

    def update(self, arm, reward, context):
        """
        Updates the internal state of the MAB after a play

        Arguments
        =========
        arm : int
            a positive integer arm id in {1, ..., self.narms}
        reward : float
            reward received from arm
        context : 1D float array, shape (self.ndims * self.narms), optional
            context given to arms
        """
        # x represents the context vector as column matrix
        x           = np.transpose(np.matrix(context[10*(arm-1):10*(arm-1)+10]))
        self.A[arm] = self.A[arm] + np.matmul(x, np.transpose(x))
        self.B[arm] = self.B[arm] + reward*x

def util_calc_reward(context, alpha, A, B):
    """
    calc reward value for an arm "i"

    Arguments
    =========
    i   :   integer
        an integer which represents arm in {1,2, ..., (self.arms-1)}
    context : 1D float array, shape (self.ndims * self.narms), optional
        context given to arms
    A   :   a numpy ndarray (2D), shape (self.narms * self.narms)
        represents (X.X_transpose + lambda*I) from linear regression
        where X is the context matrix for arms selected up till now
    B   :   a numpy ndarray (1D), shape (self.narms * 1)
        represents rewards vector (X_transponse.Y) from linear regression
    Returns
    =======
    r   :   float
        reward for arm "i"

    """
    theta = np.matmul(np.linalg.inv(A),B)
    x = np.transpose(np.matrix(context))

    r = np.matmul(np.transpose(theta), x)+\
                alpha * np.sqrt(np.matmul(np.matmul(
                    np.transpose(x),np.linalg.inv(A)),x))
    return r
## Uncomment for parallel execution
#process_pool = Pool()
