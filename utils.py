# coding: utf-8
import numpy as np

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