# AI Contest: Pacman Capture the Flag #

A contest/project *"Pacman Capture the Flag"*, based on *Berkeley Pacman Projects in AI* was launched at the University of Melbourne in October 2018. This implementation (or more specifically 'myAgentProgram.py' _File can be requested, not available publicily_ ) was ranked 10 among the 150+ entries.

*The contest rules can be found here -* [link](http://ai.berkeley.edu/contest.html)

# Learn to play ! #

* Users can create their own agent and play it against the agent program included in this repository.

  * `python capture.py -r <yourPyFile> -b myAgentProgram`
    * The agent program can be create from scratch or built upon the default program, *baselineTeam.y*

    OR

  * `python capture.py -r baselineTeam -b myAgentProgram`

    * Here baselineTeam.py will be used, which a the default agent program included in the competition.
  * OPTIONAL: The layout can be chosen from layouts folder in the repository or generated randomly. `-l RANDOM13`, where the number 13 is the seed which can be changed.


# Concepts and Techniques used in the project #

The AI agent used the following techniques and concepts to defeat the opponents.

1. [Value Iteration to decide the next move.](https://github.com/anubhav-cs/Machine-Learning-and-AI/wiki/Value-Iteration)
2. [Betweenness Centrality to determine high traffic routes in the maze.](https://github.com/anubhav-cs/Machine-Learning-and-AI/wiki/Predicting-most-likely-path-using-betweenness-centrality)
3. [Approximating the position of enemy agent using particle filter.](https://github.com/anubhav-cs/Machine-Learning-and-AI/wiki/Particle-Filter)
4. [Algorithm to find routes which can be used to trap opponents.](https://github.com/anubhav-cs/Machine-Learning-and-AI/wiki/Trap-point-finder)
5. Rule based offensive-defensive agent rotation.

Each technique is covered in detail in the project wiki [here](https://github.com/anubhav-cs/Machine-Learning-and-AI/wiki).
