# coding: utf-8
import numpy as np
import multiprocessing as mp

def epsilon_greedy_policy(qnn, observation, epsilon, env, sess, policy=None):
	"""
	Take a random action with the probability epsilon, else the best action estimated by the qnn.
	Parameters:
		qnn: qnn.QNeuralNetwork, estimator of the Q function
		observation: current state
		epsilon: probability of taking a random action
		env: environnment gym
		sess: tensorflow Session
		policy: policy to take a random action. If None, take a uniform law
	"""
	if np.random.binomial(1, epsilon):
		if policy is None:
			return env.action_space.sample()
		else:
			action = policy()
			return action
	else:
		return qnn.best_action(observation, sess)

def create_list_epsilon(n):
	"""
	Compute value of epsilon_ini for each worker. For now, it takes the value 1 with probability 0.5 
	and 0.5 with probability 0.5.
	Parameters:
		n: number of epsilons generated
	"""
	e_list = [1, 0.5]
	p = [0.5, 0.5]
	return np.random.choice(e_list, n, p=p)

	e_max = 1
	e_min = 0.01
	return [e_min + i * (e_max-e_min) / n + (e_max-e_min) / (2*n) for i in range(n)]

def initialise(input_size=4, output_size=2, n_hidden=2, hidden_size=[128, 64]):
	"""
	Initialise global variables l_theta and l_theta_minus
	Parameters:
		input_size: size of observations
		output_size: number of possible action
		n_hidden: number of hidden layers
		hidden_size: size of hidden layers
	"""
	l_theta = mp.Manager().list()
	
	shapes = [(input_size, hidden_size[0])]
	for i in range(n_hidden - 1):
		shapes.append((hidden_size[i], hidden_size[i+1]))
	shapes.append((hidden_size[-1], output_size))
	
	shapes.append((1, hidden_size[0]))
	for i in range(n_hidden - 1):
		shapes.append((1, hidden_size[i+1]))
	shapes.append((1, output_size))
	
	for i, shape in enumerate(shapes):
		l_theta.append(np.random.uniform(low=-0.01, high=0.01, size=shape))
		# l_theta[i].value = np.random.uniform(low=-0.01, high=0.01, size=shape)
		
	return l_theta