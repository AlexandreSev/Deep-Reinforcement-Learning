# coding: utf-8
import numpy as np
import multiprocessing as mp
import sys
import gym

from DRL.utils.utils import *

from DRL.slaves.asynchronenstepslave import slave_worker_n_step
from DRL.slaves.asynchrone1stepslave import slave_worker_1_step
from DRL.slaves.asynchrone1stepsarsaslave import slave_worker_1_step_sarsa
from DRL.slaves.asynchronea3cslave import slave_worker_a3c
from DRL.slaves.tester import tester_worker

from DRL.utils.settings import init



def main(nb_process, T_max=5000, t_max=5, env_name="CartPole-v0", algo="nstep",
		 model_option={"n_hidden":1, "hidden_size":[10]}, Iasyncupdate=10,
         Itarget=15, gamma=0.9, learning_rate=0.001, several_eps=True, epsilon_ini=0.9, 
         n_sec_print=10, master=False, goal=195, len_history=100, render=False, weighted=False, 
         eps_fall=50000, callback=False):
	"""
	Parameters:
		nb_process: number of slaves used in the training
	    T_max: maximum number of iterations
	    t_max: value of n in n step learning algorithm 
	    env_name: name of gym environnment
	    algo: which algorithm to use. Possible values: "nstep", "1step", "1stepsarsa"
	    model_option: dictionary, must have two keys. n_hidden defines the number of hidden layers,
	                hidden_size the size of them in the QNeuralNetwork used to estimate the reward
		Iasyncupdate: Number of steps between two updates in 1 step Q learning
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
	"""
	init(algo=algo, n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"], 
	     input_size= env_temp.observation_space.shape[0], output_size=env_temp.action_space.n)
	"""

	jobs = []
	policies = [None for i in range(nb_process)]
	if several_eps:
	    epsilons = create_list_epsilon(nb_process)
	else:
	    epsilons = [epsilon_ini for i in range(nb_process)]
	verboses = [False for i in range(nb_process)]
	if master:
	    verboses[0] = True

	if algo == "nstep":
		slave_worker=slave_worker_n_step
	elif algo == "1step":
		slave_worker=slave_worker_1_step
	elif algo == "1stepsarsa":
		slave_worker = slave_worker_1_step_sarsa
	elif algo == "a3c":
		slave_worker = slave_worker_a3c
	else:
		raise Exception("Not understood algorithm")


	for i in range(nb_process):
	    print("Process %s starting"%i)
	    job = slave_worker(T_max=T_max, model_option=model_option, env_name=env_name, 
	        policy=policies[i], epsilon_ini=epsilons[i], t_max=t_max, gamma=gamma, 
	        learning_rate=learning_rate, verbose=verboses[i], weighted=weighted, 
	        Iasyncupdate=Iasyncupdate, eps_fall=eps_fall, callback=callback, 
	        callback_name="callbacks/actor" + str(i))
	    job.start()
	    jobs.append(job)


	exemple = tester_worker(algo=algo, T_max=T_max, t_max=200, model_option=model_option, env_name=env_name, 
	                        n_sec_print=n_sec_print, goal=goal, len_history=len_history, Itarget=Itarget,
	                        render=render, weighted=weighted, callback=callback, 
	        				callback_name="callbacks/tester")
	exemple.start()
	exemple.join()

	for job in jobs:
		job.terminate()


if __name__=="__main__":
    args = sys.argv
    #np.random.seed(42)
    if len(args)>2:
        main(int(args[1]), T_max=int(args[2]), model_option={"n_hidden":2, "hidden_size":[128, 128]}, 
            render=False, master=False, env_name="CartPole-v1", goal=495, learning_rate=0.001, 
			weighted=False, algo="nstep", eps_fall=10000, callback=True)
    else:
		main(8, T_max=10000000, model_option={"n_hidden":2, "hidden_size":[128, 256]}, 
            render=False, master=False, env_name="CartPole-v0", goal=195, learning_rate=0.001, 
            weighted=False, algo="nstep", eps_fall=10000, callback=True)
