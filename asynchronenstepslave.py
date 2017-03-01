# coding: utf-8

import gym
import qnn
import multiprocessing as mp
import numpy as np
from utils import epsilon_greedy_policy

class slave_worker(mp.Process):
	"""
	The class coding a process to run a environment gym in itself and sharing gradient updates. 
	This slave uses asynchrone n step Q learning algorithm.
	"""

	def __init__(self, T_max=100000, t_max=5, gamma=0.9, learning_rate=0.001, 
	             env_name="CartPole-v0", model_option={"n_hidden":1, "hidden_size":[10]}, 
	             verbose=False, policy=None, epsilon_ini=0.9, alpha_reg=0., beta_reg=0.001, **kwargs):
		"""
		Parameters:
			T_max: maximum number of iterations
			t_max: Value of n in the n step algorithm
			gamma: depreciation of the futur
			learning_rate: learning_rate of the optimiser
			env_name: name of gym environnment
			model_option: dictionary, must have two keys. n_hidden defines the number of hidden layers,
						hidden_size the size of them in the QNeuralNetwork used to estimate the reward
			verbose: If True, each envrionnement seen by this slave will be rendered
			policy: policy used to take a random action. If None, actions are taken uniformly
			epsilon_ini: Value of epsilon at the beginning
			alpha_reg: coefficient of l1 regularisation
			beta_reg: coefficient of l2 regularisation
			kwargs: args of multiprocessing.Process
		"""
		super(slave_worker, self).__init__(**kwargs)
		self.T_max = T_max
		self.t_max = t_max
		self.gamma = gamma
		self.env = gym.make(env_name)
		self.output_size = self.env.action_space.n
		self.input_size = self.env.observation_space.shape[0]
		self.verbose = verbose
		self.epsilon_ini = epsilon_ini

		if policy is None:
		    self.policy = self.env.action_space.sample
		else:
		    self.policy = policy

		self.qnn = qnn.QNeuralNetwork(input_size=self.input_size, output_size=self.output_size, 
				n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"], 
				learning_rate=learning_rate, alpha_reg=alpha_reg, beta_reg=beta_reg)
		    
	    
	def run(self):
		"""
		Run the worker and launch the n step algorithm
		"""
		import tensorflow as tf
		global T, l_theta, l_theta_minus

		self.qnn.initialisation()

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		self.qnn.read_value_from_theta(self.sess)
		self.qnn.read_value_from_theta_minus(self.sess)

		epsilon = self.epsilon_ini
		nb_env = 0

		observation = self.env.reset()

		while T.value<self.T_max:

		    t = 0
		    t_init = t
		    done = False

		    observation_batch = observation.reshape((1, -1))

		    reward_batch = []
		    action_batch = []

		    self.qnn.read_value_from_theta(self.sess)


		    while (not done) & (t-t_init<=self.t_max):
		    
		        if self.verbose:
		            self.env.render()
		            if T.value%5000 == 0:
		                print("T = %s"%T.value)

		        self.qnn.read_value_from_theta(self.sess)

		        action = epsilon_greedy_policy(self.qnn, observation, epsilon, self.env, 
		        								self.sess, self.policy)

		        observation, reward, done, info = self.env.step(action) 

		        reward_batch.append(reward)
		        action_batch.append(action)
		        observation_batch = np.vstack((observation.reshape((1, -1)), observation_batch))
		      

		        if done:
		            nb_env += 1
		            observation = self.env.reset()
		        
		        with T.get_lock():
		            T.value += 1
		        
		        t += 1

		        if epsilon>0.01:
		            epsilon -= (self.epsilon_ini - 0.01)/200000
		    
		    if done:
		        R = 0
		    else:
		        R = self.qnn.best_reward(observation, self.sess)

		    true_reward = []
		    for i in range(t - 1, t_init - 1, -1):
		        R = reward_batch[i] + self.gamma * R
		        true_reward.append(R)

		    action_batch.reverse()
		    action_batch_multiplier = np.eye(self.output_size)[action_batch].T
		    
		    y_batch_arr = np.array(true_reward).reshape((-1, 1))

		    shuffle = range(len(y_batch_arr))
		    np.random.shuffle(shuffle)
		    
		    self.qnn.read_value_from_theta(self.sess)
		    
		    feed_dict = {self.variables_dict["input_observation"]: observation_batch[:-1, :][shuffle, :],
		                 self.variables_dict["y_true"]: y_batch_arr[shuffle, :], 
		                 self.variables_dict["y_action"]: action_batch_multiplier[:, shuffle]}
		    self.sess.run(self.train_step, feed_dict=feed_dict)


		    self.qnn.assign_value_to_theta(self.sess)

		return






