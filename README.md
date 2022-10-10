# Truncated Quantile Critic + D2RL - Bipedal Walker 
Submitted as part of the degree of Msci Natural Sciences (3rd year) to the Board of Examiners in the Department of Computer Sciences, Durham University. 
This summative assignment was assessed and marked by the professor of the module in question:

## Grade: 1st - 100/100, 1st in year (of 136 students).
> "Convergence: The agent has perfect convergence and is ranked at the very top of the class. It achieves
the highest scores extremely quickly within very few episodes. 
> Sophistication and mastery of the domain: This is a perfect reinforcement learning submission. You’ve clearly done a large amount of independant
study and have outstanding grasp of the underpinning theory related to overestimation bias, exploration exploitation,
and state-of-the-art methods to improve sample efficiency. The paper desmonstrates an outstanding
level of rigorour in terms of the scientific approach to hyperparameter optimisation, and the results
speak for themselves. 
> Intuition of the video: The video is extremely intuitive and has clearly learnt a near
optimal policy to run quickly. There is no stuttering or signs of continued exploration, so the agent has
converged on exploiting a policy similar to that of Usain Bolt." - Dr Chris G. Willcocks

## Abstract:
This paper surveyed and evaluated four different approaches (Twin Deep Deterministic Policy, Soft Actor Critic, Augmented Random Search and Truncated Quantile Critic) for the Bipedal Walker environment and distinguished Truncated Quantile Critic as the most advantageous. It focuses on performing a robust and comprehensive hyperparameter sweep (as initial trials indicated that all agents were very sensitive to small hyperparameter changes) to find the hyperparameters most suitable for the Bipedal Walker environment. It then augmented the neural architecture by incorporating findings from D2RL. The resultant agent is able to solve both the BipedalWalker and BipedalWalkerHardcore environments.

### Video of Agent:
  >![Gifdemo2](https://github.com/Lauren-Stumpf/Reinforcement_Learning_Coursework/blob/main/bipedal_walker_score%3D330.gif)

## Contents:
* agent_code.py - Implemented TQC + D2RL
* agent_paper.pdf - Paper outlining methodology
* bipedal_hardcore_score=314.gif - Video of the agent completing the challenging bipedal hardcore environment
* bipedal_walker_score=330.gif - Video of the agent completing the bipedal environment  


## Results:
> The left chart shows my personalised log data relative to everyone else giving me a convergence rank of 100.0%. The right chart shows the convergence ranks of all agents, with monotonically increasing high scores - otherwise the visualisation is too noisy to see anything.
  > ![Training graph](convergent_graph.png?raw=true "Convergence Graph")


  


