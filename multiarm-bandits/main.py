################################################################################
#                                                                              #
# Statistical Machine Learning & AI                                            #
#                                                                              #
# Problem:		Multi-arm bandits                                              #
#                                                                              #
# Author:       Anubhav Singh                                                  #
#                                                                              #
# References:                                                                  #
#   1. offlineEvaluate:                                                        #
#      a.) https://arxiv.org/pdf/1003.0146.pdf                                 #
#      Contextual-Bandit Approach to Personalized News Article Recommendation  #
#      b.) https://arxiv.org/pdf/1003.5956.pdf                                 #
#      Unbiased Offline Evaluation of Contextual-bandit-based -                #
#                               News Article Recommendation Algorithms         #
#                                                                              #
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from eps_greedy import EpsGreedy
from ucb import UCB
from lin_ucb import LinUCB
from kernel_ucb import KernelUCB
from sklearn.metrics.pairwise import rbf_kernel


def offlineEvaluate(mab, arms, rewards, contexts, nrounds=None):
    """
    Offline evaluation of a multi-armed bandit
    Implementation of Algorithm 3 in the paper:-
        - https://arxiv.org/pdf/1003.0146.pdf

    Arguments
    =========
    mab : instance of MAB

    arms : 1D int array, shape (nevents,)
        integer arm id for each event

    rewards : 1D float array, shape (nevents,)
        reward received for each event

    contexts : 2D float array, shape (nevents, mab.narms*nfeatures)
        contexts presented to the arms (stacked horizontally)
        for each event.

    nrounds : int, optional
        number of matching events to evaluate `mab` on.

    Returns
    =======
    out : 1D float array
        rewards for the matching events
    """

    mab_arm = None
    tr_index = 1
    reward_out = []
    history = []
    for arm, reward, context in zip(arms, rewards, contexts):
        mab_arm = mab.play(tr_index,context)
        if arm != mab_arm:
            continue
        else:
            tr_index += 1
            mab.update(arm, reward, context)
            reward_out.append(reward)
        mab_arm = None
        if nrounds!= None and tr_index > nrounds:
            break
    return reward_out

def loadDataset(file):
    """
    Function to load the data-set into the main memory from text file.

    Arguments
    =========
    file        :   {file path}/filenme

    Returns
    =======
    arms        :   1D int array
        arms for the matching events,
        in an array indexed by event/row number in the dataset
    rewards     :   1D float array
        rewards for the matching events
    contexts    :   1D float array
        context for the matching events

    """

    f = open(file, "r")
    rows  = [[(int)(x) for x in line.rstrip('\n').split()] for line in f]
    arms = [row[0] for row in rows]
    rewards = [(float)(row[1]) for row in rows]
    contexts = [[ (float)(val) for val in row[2:]] for row in rows]
    return arms, rewards, contexts

def alphaGridSearch():
    """
    Function to perform grid-search to find optimal alpha value for Linear UCB

    """

    step_size = 0.01
    count_loop = 200
    alpha_grid = []
    alpha_grid.append(0)

    for i in range(count_loop):
        alpha_grid.append(alpha_grid[i]+step_size)

    results = []
    for alpha in alpha_grid:
        mab = LinUCB(10, 10, alpha)
        results_Test = offlineEvaluate(mab, arms, rewards, contexts, 800)
        results.append(np.mean(results_Test))

    max_val = max(results)

    print("Best value of alpha = ",alpha_grid[results.index(max_val)])
    plt.plot(alpha_grid, results, label= "LinUCB")
    plt.xlabel("$\\alpha$")
    plt.ylabel("$T^{-1}\sum_{1}^{T} r_{T,a}$")
    plt.legend()


if __name__ == '__main__':
    """
    The analysis of the algorithms on the dataset.

    """
    #Load the dataset
    arms, rewards, contexts = loadDataset("dataset.txt")

    # Evaluate algorithm's effectiveness in terms of average rewards at the end
    mab = EpsGreedy(10, 0.05)
    results_EpsGreedy = offlineEvaluate(mab, arms, rewards, contexts)
    print('EpsGreedy average reward', np.mean(results_EpsGreedy))

    mab = UCB(10, 1.0, 0.1)
    results_UCB = offlineEvaluate(mab, arms, rewards, contexts, None)
    print('UCB average reward', np.mean(results_UCB))

    mab = LinUCB(10, 10, 1.0)
    results_LinUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('LinUCB average reward', np.mean(results_LinUCB))

    "NOTE: Kernel UCB is resource intensive and may take much longer to finish"
    mab = KernelUCB(10, 10, 0.5, 0.12, rbf_kernel)
    results_KernelUCB = offlineEvaluate(mab, arms, rewards, contexts, 800)
    print('KernelUCB average reward', np.mean(results_KernelUCB))

    # Plot the average-reward graph over time
    plot_EpsGreedy  = []
    plot_UCB        = []
    plot_LinUCB     = []
    plot_KernelUCB  = []

    for T in range(800):
        plot_EpsGreedy.append(np.mean(results_EpsGreedy[:T+1]))
        plot_UCB.append(np.mean(results_UCB[:T+1]))
        plot_LinUCB.append(np.mean(results_LinUCB[:T+1]))
        plot_KernelUCB.append(np.mean(results_KernelUCB[:T+1]))
    plt.plot(range(1, 801), plot_EpsGreedy, label= "EpsGreedy")
    plt.plot(range(1, 801), plot_UCB, label ="UCB")
    plt.plot(range(1, 801), plot_LinUCB, label = "LinUCB")
    plt.plot(range(1, 801), plot_KernelUCB, label= "KernelUCB")

    plt.xlabel("$T$")
    plt.ylabel("$T^{-1}\sum_{1}^{T} r_{T,a}$")
    plt.legend()
