# DeepReinforcementLearning
This repository contains the code developed during the group project as part of the assessment in the Reinforcement Learning course at Uppsala University. The project consists of a Deep Q-Network (DQN) for playing Atari games, in particular Pong, using Reinforcement Learning.

Instructions on usage:

- To train or evaluate the agent on an environment, the testing.py file can be run. The script has a main, where one needs to choose the environment and whether to train or evaluate.

- In the file train.py, one needs to define paths for saving plots and models. Lines 44 and 152 needs to be hard-coded.

- In the file testing.py, a path needs to be defined to load a saved model. Line 29 needs to be hard-coded.

- In file train.py, two variables need to be defined, namely 'evaluate_freq' and 'evaluation_episodes'. By default they are set to 25 and 4, respectively.

Authors:
- Hardy Hasan
- Marcello Vendruscolo