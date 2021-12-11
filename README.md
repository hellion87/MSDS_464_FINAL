# MSDS_464_FINAL

#### This repo is used to store the reinforcement learning (RL) project for MSDS_464 (Team 2).


________

##### We use RL to train an agent to play chess.

##### The docs folder contains LaTeX documents for our writeup
* /docs - Contains the PDF and tex files.
* /notebooks - Contain the Jupyter notebooks for our experiments.

___________

## Abstract

Chess is one of the most studied domains of artificial intelligence. Unfortunately, the massive branching factor and state-space make an exhaustive search intractable. We introduce multiple methods to build a chess-playing algorithm to overcome these challenges. First, we evaluate a Monte Carlo Search Tree (MCST) agent that uses an Upper Confidence Bounds for Trees (UCT) algorithm for the tree policy and acts randomly for the rollout policy. This algorithm performs poorly when iterating 50,000 times per move, making short-sighted moves because of the shallow search tree. Second, we implement an actor-critic model that uses convolutional networks to generalize by training the agent with 1000 games of self-play. This agent plays better but is still poor. Last, we expand the AlphaGo Zero algorithm by implementing an alternative training target that incorporates temporal difference learning ideas. In our approach, we compute the Q value of each state through repeated simulations from that node. In contrast, the standard AlphaGo Zero algorithm back-propagates the episode reward to estimate the value of state/action pairs. This agent performs best, beating our AlphaGo Zero agent 31 games, losing 19 games, and drawing 50 games. 
