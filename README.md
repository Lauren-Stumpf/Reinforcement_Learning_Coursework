# Truncated Quantile Critic + D2RL - Bipedal Walker 
Submitted as part of the degree of Msci Natural Sciences (3rd year) to the Board of Examiners in the Department of Computer Sciences, Durham University. 
This summative assignment was assessed and marked by the professor of the module in question:

## Grade: 1st - 100/100, 1st in year (of 136 students).
> "Convergence: The agent has perfect convergence and is ranked at the very top of the class. It achieves
the highest scores extremely quickly within very few episodes. 
> Sophistication and mastery of the domain: This is a perfect reinforcement learning submission. You’ve clearly done a large amount of independant
study and have outstanding grasp of the underpinning theory related to overestimation bias, explorationexploitation,
and state-of-the-art methods to improve sample efficiency. The paper desmonstrates an outstanding
level of rigorour in terms of the scientific approach to hyperparameter optimisation, and the results
speak for themselves. 
> Intuition of the video: The video is extremely intuitive and has clearly learnt a near
optimal policy to run quickly. There is no stuttering or signs of continued exploration, so the agent has
converged on exploiting a policy similar to that of Usain Bolt." - Dr Chris G. Willcocks

## Abstract:
This paper surveyed and evaluated four different approaches (Twin Deep Deterministic Policy, Soft Actor Critic, Augmented Random Search and Truncated Quantile Critic) for the Bipedal Walker environment and distinguished Truncated Quantile Critic as the most advantageous. It focuses on performing a robust and comprehensive hyperparameter sweep (as initial trials indicated that all agents were very sensitive to small hyperparameter changes) to find the hyperparameters most suitable for the Bipedal Walker environment. It then augmented the neural architecture by incorporating findings from D2RL. The resultant agent is able to solve both the BipedalWalker and BipedalWalkerHardcore environments.

## Contents:
* 
* An example video of the (partially) converged agent playing Gravitar, achieving a score of 5050. This episode was not an anomaly nor an unusual result (see graph).
* Training logs for statistical analysis or plotting of data.
* A graph of the training scores over time. 
* 


## Results:

### Demo video (taken from my [portfolio page](https://github.com/shadowbourne)):
  >![Gifdemo2](https://user-images.githubusercontent.com/18665030/136660504-c89f9c89-41d3-4070-982f-23473bda3fcb.gif)
  >
  > We were tasked with creating a Reinforcement Learning agent to play the notoriously difficult Gravitar from the Atari-57 suite. I therefore decided to look for the current state-of-the-art Reinforcement Learning model for Atari ([R2D2](https://openreview.net/pdf?id=r1lyTjAqYX)) and re-created it to the best I could with my limited hardware. I produced the best agent in the class, and my convergence graph was used as exemplar feedback to the cohort.


### Training graph of mean scores (rolling average over past 100 episodes):
![Training graph](training-graph.png?raw=true "Training graph")
**Note:** This graph was selected as the best in the cohort and included in the class feedback as one of two exemplar graphs.

As is clear from the graph, the agent has not yet fully converged.

