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
import numpy as np

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
        self.narms = narms
        self.ndims = ndims
        self.alpha = alpha
        self.arm_at_t = {}
        self.A = {}
        self.B = {}
        self.theta ={}
        self.num_times_arm = {}
        for i in range(self.narms):
            # Initialize A as Identity Matrix
            self.A[i+1] = np.identity(ndims)
            # Initialize B as Zero Vector
            self.B[i+1] = np.zeros((ndims,1))

    def play(self, tround, context):
        """
        Returns a selected arm
        =======
        self.arm_at_t[tround] : int
            the positive integer arm id for this round

        """

        # Enables multiple play calls for same round number (in case needed)
        # - by resetting the counters updated by previous play call.
        if tround in self.arm_at_t.keys():
            tround_repeat = True
            self.num_times_arm[self.arm_at_t[tround]]-=1
        else:
            tround_repeat = False

        p ={}
        t = []
        for i in range(self.narms):
            self.theta[i+1] = np.matmul(np.linalg.inv(self.A[i+1]),self.B[i+1])
            x = np.transpose(np.matrix(context[10*i:10*i+10]))
            p[i+1] = np.matmul(np.transpose(self.theta[i+1]), x)+\
            self.alpha * np.sqrt(np.matmul(np.matmul(np.transpose(x),np.linalg.inv(self.A[i+1])),x))
            t.append(p[i+1])
        max_val = max(p[k] for k in p.keys())
        list_max = []
        for k in p.keys():
            if p[k] == max_val:
                list_max.append(k)

        self.arm_at_t[tround] = np.random.choice(list_max)

        self.num_times_arm[self.arm_at_t[tround]] = self.num_times_arm.setdefault(self.arm_at_t[tround],0)+1

        return self.arm_at_t[tround]

    def update(self, arm, reward, context):
        """
        Updates the internal state of the MAB after a play

        """
        x = np.transpose(np.matrix(context[10*(arm-1):10*(arm-1)+10]))
        self.A[arm] = self.A[arm] + np.matmul(x, np.transpose(x))
        self.B[arm] = self.B[arm] + reward*x
