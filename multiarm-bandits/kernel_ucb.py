################################################################################
#                                                                              #
# Statistical Machine Learning & AI                                            #
#                                                                              #
# Problem:  Contextual bandits                                                 #
#                                                                              #
# Author:   Anubhav Singh                                                      #
#                                                                              #
# References:                                                                  #
#   1. Kernel UCB: https://arxiv.org/ftp/arxiv/papers/1309/1309.6869.pdf       #
#       "Finite-Time Analysis of Kernelised Contextual Bandits",               #
#           Michal Valko, Nathan Korda, R ́emi Munos,                          #
#           Ilias Flaounas, Nello Cristianini                                  #
#                                                                              #
################################################################################

from mab import MAB
import numpy as np
from numpy.linalg import inv

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

class KernelUCB(MAB):
    """
    Kernelised contextual multi-armed bandit (Kernelised LinUCB)
    Implementation of Kernel UCB ALgorithm from paper:-
        Algorithm-1, https://arxiv.org/ftp/arxiv/papers/1309/1309.6869.pdf

    Arguments
    =========
    narms : int
        number of arms

    ndims : int
        number of dimensions for each arm's context

    gamma : float
        positive real explore-exploit parameter

    eta : float
        positive real explore-exploit parameter

    kern : callable
        a kernel function from sklearn.metrics.pairwise
    """
    def __init__(self, narms, ndims, gamma, eta, kern):
        self.narms = narms
        self.ndims = ndims
        self.gamma = gamma
        self.eta = eta
        self.kern = kern
        self.context_at_t = {}
        self.reward_at_t = []
        self.last_round = 0
        self.u = np.transpose(np.zeros((1,narms)))
        self.y = []
        self.k_gamma = 0.01

    def play(self, tround, context):
        """
        Play a round

        Arguments
        =========
        tround : int
            positive integer identifying the round

        context : 1D float array, shape (self.ndims * self.narms),
        optional context given to the arms

        Returns
        =======
        arm : int
            the positive integer arm id for this round
        """

        if tround == 1:
            # by default play arm 1
            self.u[0][0] = 1
        else:
            for i in range(self.narms):
                x = np.matrix(context[10*(i):10*(i)+10])
                k_arr = []
                for j in range(1,tround):
                    k_arr.append(self.kern(x, self.context_at_t[j], self.k_gamma)[0][0])

                k_i = np.matrix(k_arr)

                # standard deviation of the gaussian distribution
                sigma = np.sqrt(self.kern(x,x, self.k_gamma)[0][0] - \
                        np.linalg.det(np.matmul(np.matmul
                        (k_i,self.K_inv),np.transpose(k_i))))

                # mean reward for arm i
                self.u[i] =  np.linalg.det(np.matmul(np.matmul(k_i,\
                                self.K_inv),self.y)) + \
                                (self.eta*sigma)/np.sqrt(self.gamma)

        arm = argmax_rand(self.u)+1
        self.context_at_t[tround] = np.matrix(context[10*(arm-1):10*(arm-1)+10])
        self.last_round = tround

        return arm

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
        a = np.matrix(context[10*(arm-1):10*(arm-1)+10])
        self.reward_at_t.append(reward)

        # Create reward column vector 'y'
        self.y = np.transpose(np.matrix(self.reward_at_t))
        k_arr = []

        # Calculation of Kernel Matrix Inverse
        ## Kernel matrix increases in size by 1, at each time step
        if self.last_round == 1:
            self.K_inv = inv(
                np.matrix([self.kern(a,a, self.k_gamma)[0][0]+self.gamma]))
        else:
            for i in range(1,self.last_round):
                k_arr.append(
                    self.kern(a, self.context_at_t[i], self.k_gamma)[0][0])

            b = np.matrix(k_arr)

            K_22 = inv(self.kern(a,a, self.k_gamma)[0][0] +\
                                self.gamma - np.matmul(
                                    np.matmul(b,self.K_inv),np.transpose(b)))
            K_11 = self.K_inv + (np.linalg.det(K_22)*\
                                np.matmul(np.matmul(self.K_inv,
                                    np.transpose(b)), np.matmul(b,self.K_inv)))
            K_12 = -np.linalg.det(K_22)*np.matmul(self.K_inv,np.transpose(b))

            K_21 = -np.linalg.det(K_22)*np.matmul(b,self.K_inv)

            m = np.concatenate((np.array(K_11), np.array(K_12)),axis=1)
            n = np.concatenate((np.array(K_21), np.array(K_22)),axis=1)

            self.K_inv = np.concatenate((m,n), axis=0)
