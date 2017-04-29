# coding: utf-8
from main import main

#number of learners
n_learners = 8

#Maximum time step
T_max = 10000000

#number of hidden layers in the neural networks
n_hidden = 2

#number of units in the hidden layers
hidden_size = [64, 128]

#Shall the tester render its actions ?
render = False

#Shall the first learner render its action ?
master = False

#Name of the environnement ? Environnements can be find on gym.openai.com
env_name = "CartPole-v1"

#Goal to complete the environnement
goal = 475

#Which algo do you want to use ? Must be in ["1step", "1stepsarsa", "nstep", "a3c"]
algo = "a3c"

#Value of n in nstep algorithme
t_max = 5

#Number of time step for epsilon to reach its final value
eps_fall = 100000

#Shall we create callbacks ? 
callback = True

#Number of timesteps between two updates of the target network
Itarget = 100

#Number of timestep an action is repeted. Can save some computational cost.
action_replay = 1

#Shall we reset the learning rate and espsilon ? It slows the training, but makes it more stable.
reset = True

#Shall we use precomputed weights? Be carefull, hidden_size and n_hidden must be the same
warmstart = False

#Where are the weights ?
weights_path = "./checkpoints/Try_76/best_weights"




main(n_learners, T_max=T_max, model_option={"n_hidden":n_hidden, "hidden_size":hidden_size}, 
            render=render, master=master, env_name=env_name, goal=goal, algo=algo, 
            eps_fall=100000, callback=True, Itarget=100, action_replay=1, reset=reset, 
            warmstart=warmstart, weights_path=weights_path, t_max=t_max)