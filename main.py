# coding: utf-8
import multiprocessing as mp
import gym
import sys
from utils import *
from asynchronenstepslave import slave_worker
from tester import tester_worker
from settings import init


def main(nb_process, T_max=5000, t_max=5, env_name="CartPole-v0", 
		 model_option={"n_hidden":1, "hidden_size":[10]}, 
         Itarget=15, gamma=0.9, learning_rate=0.001, several_eps=True, epsilon_ini=0.9, 
         n_sec_print=10, master=False, goal=195, len_history=100, render=False):
	"""
	Parameters:
		nb_process: number of slaves used in the training
	    T_max: maximum number of iterations
	    t_max: value of n in n step learning algorithm 
	    env_name: name of gym environnment
	    model_option: dictionary, must have two keys. n_hidden defines the number of hidden layers,
	                hidden_size the size of them in the QNeuralNetwork used to estimate the reward
	    Itarget: number of iterations between two updates of theta minus
	    gamma: depreciation of the futur
		learning_rate: learning_rate of the optimiser
		several_eps: if True, epsilon_ini of the workers will be created using utils.create_list_epsilon
					else, they are all set to epsilon_ini
			epsilon_ini: not used if several_eps. Else, initialisation of epsilons in the slave worker
	    n_sec_print: Number of seconds between two prints
	    master: If True, a worker will show its work
	    goal: Value of reward to considered the game as solved
	    len_history: number of episodes to test the algorithm
	    render: If True, environments of the tester will be rendered
	    kwargs: args of multiprocessing.Process
	"""

	env_temp = gym.make(env_name)

	init(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"], 
	     input_size= env_temp.observation_space.shape[0], output_size=env_temp.action_space.n)

	jobs = []
	policies = [None for i in range(nb_process)]
	if several_eps:
	    epsilons = create_list_epsilon(nb_process)
	else:
	    epsilons = [epsilon_ini for i in range(nb_process)]
	verboses = [False for i in range(nb_process)]
	if master:
	    verboses[0] = True
	for i in range(nb_process):
	    print("Process %s starting"%i)
	    job = slave_worker(T_max=T_max, model_option=model_option, env_name=env_name, 
	        policy=policies[i], epsilon_ini=epsilons[i], t_max=t_max, gamma=gamma, 
	        learning_rate=learning_rate, verbose=verboses[i])
	    job.start()
	    jobs.append(job)


	exemple = tester_worker(T_max=T_max, t_max=500, model_option=model_option, env_name=env_name, 
	                        n_sec_print=n_sec_print, goal=goal, len_history=len_history, Itarget=Itarget,
	                        render=render)
	exemple.start()
	exemple.join()


if __name__=="__main__":
    args = sys.argv
    if len(args)>2:
        main(int(args[1]), T_max=int(args[2]), model_option={"n_hidden":2, "hidden_size":[128, 128]}, 
            render=True, master=False, env_name="CartPole-v0", goal=1000)
    else:
        main(3, 50000)
