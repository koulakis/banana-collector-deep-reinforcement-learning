# Banana collector

## Introduction
This is a solution for the first project of the [Udacity deep reinforcement learning course](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). It includes code for training an agent and for using it in a simulation environment.

## Example agent
The giff shows the behavior of an agent trained with DQN in this codebase. The agent parameters can be found under `experiments/final_comparison_dqn/checkpoint.pth`.
![Agent test run](artifacts/screencast_unity_edited.gif)

## Problem description
The agent is placed in a room with bananas and its goal is to collect as many fresh (yellow) bananas as possible during a fixed amount of time while avoiding rotten bananas (blue).

- Rewards:
  - +1 for each yellow banana collected
  - -1 for each blue banana collected
- Input state:
  - 37 dimesions with the agent's velocity and ray-based perception of the environment
- Actions:
  - 0: forward
  - 1: backward
  - 2: left
  - 3: right
- Goal:
  - Get an average score of at least 13 over 100 consecutive episodes

## Solution
The problem is solved by training an agent using a deep Q-learning architecture with some improvements including
double DQL, prioritized experience replay and dueling DQN. For more details look in the (corresponding report)[<tbd>]. 

## Setup project
To setup the project follow those steps:
- Provide an environment with `python 3.6.x` installed, ideally create a new one with e.g. pyenv or conda
- Clone and install the project: 
```
git clone git@github.com:koulakis/banana-collector-deep-reinforcement-learning.git
cd banana-collector-deep-reinforcement-learning
pip install .
```
- Create a directory called `udacity_custom_unity_agent` under the root of the project and download and extract there
  the environment compatible with your architecture. You can find the (download links here)[https://github.com/udacity/deep-reinforcement-learning/tree/master/p1_navigation#getting-started].
- Install a version of pytorch compatible with your architecture. The version used to develop the project was 1.5.0.
e.g. `pip install pytorch==1.5.0`

To check that everything is setup properly, run the following test which loads an environment and runs a random agent:
`python scripts/test_agent_in_environment.py --agent-type random`

## Training and testing the agent
The project comes along with some pre-trained agents, scripts to test them and train your own.

### Scripts
- `train_agent.py`: This one is used to train an agent. The parameter `experiment-name` is used to name your agent and
    the script will create by default a directory under `experiments` with the same name. The trained agent parameters
    will be saved there in the end of the training along with timeseries of the scores measured during training. Here is
    an example call:
    ```python scripts/train_agent.py --device 'cuda:0' --double-dqn --experiment-name my_fancy_agent_using_double_dqn```
    
- `test_agent_in_environment`: This script can be used to test an agent on a given environment. As mentioned above, one
can access the saved agent models inside the sub-folders of `experiments`. An example usage:
    ```python scripts/test_agent_in_environment.py --agent-parameters-path experiments/dqn_training/checkpoint.pth```
    
- `evaluate_agent`: This script is used to evaluate the agent on a number of episodes. The agent is not trained further
during the evaluation and the exploration rate is set to 0.
    
### Pre-trained models
Under the `experiments` directory there are several pre-trained agents one can used to run in the environment. All of
them have solved the environment so they are expected to have a relative intelligent behaviour.

## References
Given that this project is an assignment of an online course, it has been influenced heavily by code provided by
Udacity and several mainstream publications. Below you can find some links which can give some broader context.

### Codebases
1. Most of the simulation setup comes from (this notebook)[https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/Navigation.ipynb]
1. The architecture of the agent and the training loop was influenced by a (similar project in the course)[https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution]
1. The unity environment created by Udacity is a direct copy (from here)[https://github.com/udacity/deep-reinforcement-learning/tree/master/python]
1. Some implementation details (e.g. for the dueling DQN architecture) were inspired from the (ReAgent project)[https://github.com/facebookresearch/ReAgent]
 
### Publications
The following publications were either directly used to build improvements of the original DQN algorithm or provided
inspiration for setting hyper parameters and understanding the problem.

1. *Human-level control through deep reinforcement learning*. Mnih, V., Kavukcuoglu, K., Silver, D. et al. . Nature 518, 529–533. 2015.
1. *Prioritized Experience Replay*. Tom Schaul and John Quan and Ioannis Antonoglou and David Silver. arXiv:1511.05952. 2015.
1. *Deep Reinforcement Learning with Double Q-Learning*. Ziyu Wang and Tom Schaul and Matteo Hessel and Hado van Hasselt and Marc Lanctot and Nando de Freitas. arXiv:1511.06581. 2015.
1. *Deep Reinforcement Learning with Double Q-learning*. Hado van Hasselt and Arthur Guez and David Silver. arXiv:1509.06461. 2015.
1. *Rainbow: Combining Improvements in Deep Reinforcement Learning*. Matteo Hessel and Joseph Modayil and Hado van Hasselt and Tom Schaul and Georg Ostrovski and Will Dabney and Dan Horgan and Bilal Piot and Mohammad Azar and David Silver. arXiv:1710.02298. 2017.