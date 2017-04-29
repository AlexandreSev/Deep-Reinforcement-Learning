# Deep-Reinforcement-Learning

This repo is a implementation of asynchronous algorithms like they are described in the paper "Asynchronous Methods for Deep Reinforcement Learning" avaible here:

https://arxiv.org/pdf/1602.01783.pdf

## Demonstration

A user-friendly demonstration is available in the files demo_training.py and dome_trained.py, where parameters are explained.

- demo_training.py is configured by default to show the training of A3C on the environment CartPole with 8learners.

- demo_trained.py is configured to show the result of a fully trained A3C on this same environment.

If you launch the first demonstration, you can follow the training with two tools.

- First, you can use the dashboard present in this repo to visualize some graphics on each learner and the reward obtained by the tester. You just have to change the path of the callbacks in the third cell.
Be carefull, the first plot requiere a lot of memory.

- Then, you can use Tensorboard to visualize the loss, the learning rate and the distribution of the weights of the neural network. 
You can also visualize a graph representing the tensorflow graph of this neural network. 

To open Tensorboard, you have to launch from the terminal the command line: "tensorboard --logdir=path" where path is the path where 
the summaries are. By default, they are in this repo in "./callbacks/summaries".
