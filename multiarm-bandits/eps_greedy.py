################################################################################
#                                                                              #
# Statistical Machine Learning & AI                                            #
#                                                                              #
# Problem:		Multi-arm bandits                                              #
#                                                                              #
# Author:       Anubhav Singh                                                  #
#                                                                              #
################################################################################

from mab import MAB
import numpy as np

def argmax_rand(a):
    """
    Random-pick an index from the set of
    indices corresponding to a's max value.

    """
    return np.random.choice(np.flatnonzero(a == a.max()))

class EpsGreedy(MAB):
    """
    Epsilon-Greedy multi-armed bandit

    Arguments
    =========
    narms : int
        number of arms

    epsilon : float
        explore probability

    Q0 : float, optional
        initial value for the arms
    """

    def __init__(self, narms, epsilon, Q0=np.inf):
        """
        Initialize the instance variables
        """
        self.narms = narms
        self.epsilon = epsilon
        self.Q0 = Q0
        self.Q = np.full(narms, Q0)
        self.arm_at_t = {}
        self.reward_at_t = []
        self.num_times_arm = {}

    def play(self, tround, context=None):
        """
        Returns a selected arm
        =======
        self.arm_at_t[tround] : int
            the positive integer arm id for this round

        """

        # Rejects multiple play calls for same round number
        # - by resetting the counters updated by previous call.
        if tround in self.arm_at_t.keys():
            tround_repeat = True
            self.num_times_arm[self.arm_at_t[tround]]-=1
        else:
            tround_repeat = False

        # Explore vs Exploit:
        exploit \
        = np.random.choice([True,False], 1, \
                           replace=True, p=[1 -self.epsilon, self.epsilon])

        if exploit:
            # Random tie-break against all index
            # corresponding to max value of Q
            self.arm_at_t[tround] = argmax_rand(self.Q)
        else:
            # Random tie-break
            self.arm_at_t[tround] \
                = np.random.choice(range(self.narms))

        # Update the counter for number of times an arm was pulled
        self.num_times_arm[self.arm_at_t[tround]] \
            = self.num_times_arm.setdefault(self. \
                                            arm_at_t[tround],0)+1

        return self.arm_at_t[tround]+1

    def update(self, arm, reward, context=None):
        """
        Updates the internal state of the MAB after a play

        """
        arm = arm - 1
        self.reward_at_t.append(reward)
        if self.Q[arm]==np.inf:
            self.Q[arm] = reward
        else:
            self.Q[arm] = ((self.Q[arm]*(self.num_times_arm[arm] \
                                -1))+reward)/self.num_times_arm[arm]
