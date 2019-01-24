# Analysis of recommendation engine using multi-arm bandits(MAB) and contextual bandits algorithms #

![Graph-Plot of relative performance](https://github.com/anubhav-cs/Machine-Learning-and-AI/blob/master/multiarm-bandits/images/foo.png)

The following algorithms were implemented:

1. Epsilon Greedy
2. Upper confidence bounds
3. Linear UCB
4. Kernel UCB

Furthermore, these algorithms were tested against the data-set, 10,000 lines (i.e., rows) corresponding to distinct site visits by users—events in the language of this part. Data-set was taken from university of melbourne's, statistical machine learning course.

* Each row comprises 102 space-delimited columns of integers:

  – Column 1: The arm played by a uniformly-random policy out of 10 arms (news articles);

  – Column 2: The reward received from the arm played—1 if the user clicked 0 otherwise; and

  – Columns 3–102: The 100-dim flattened context: 10 features per arm (incorporating the content of
the article and its match with the visiting user), first the features for arm 1, then arm 2, etc. up to
arm 10.


# References: #

Included in relevant python code files, but overall refers to the following papers.


* Lihong Li, Wei Chu, John Langford, Robert E. Schapire, ‘A Contextual-Bandit Approach to Personalized News Article Recommendation’, in Proceedings of the Nineteenth International Conference on World Wide Web (WWW 2010), Raleigh, NC, USA, 2010. <https://arxiv.org/pdf/1003.0146.pdf>

* Lihong Li, Wei Chu, John Langford, and Xuanhui Wang.  ‘Unbiased offline evaluation of contextual bandit-based news article recommendation algorithms.’  In Proceedings of the Fourth ACM International Conference on Web Search and Data Mining (WSDM’2011)
, pp.  297-306. ACM, 2011. <https://arxiv.org/pdf/1003.5956.pdf>

* Michal Valko, Nathan Korda, R ́emi Munos, Ilias Flaounas, and Nello Cristianini, ‘Finite-time analysis of kernelised contextual bandits.’  In Proceedings of the Twenty-Ninth Conference on Uncertainty in Artificial Intelligence (UAI’13), pp.  654-663.  AUAI Press, 2013. <https://arxiv.org/ftp/arxiv/papers/1309/1309.6869.pdf>
