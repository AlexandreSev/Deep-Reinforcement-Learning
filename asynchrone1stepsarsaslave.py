# coding: utf-8

import gym
import qnn
import multiprocessing as mp
import numpy as np
from utils import epsilon_greedy_policy
import settings

class slave_worker_1_step_sarsa(mp.Process):
	"""
	The class coding a process to run a environment gym in itself and sharing gradient updates. 
	This slave uses asynchrone 1-step Q learning algorithm.
	"""

	def __init__(self, T_max=100000, t_max=5, gamma=0.9, learning_rate=0.001, Iasyncupdate=10,
				 env_name="CartPole-v0", model_option={"n_hidden":1, "hidden_size":[10]}, 
				 verbose=False, policy=None, epsilon_ini=0.9, alpha_reg=0., beta_reg=0.001, 
				 weighted=False, **kwargs):
		"""
		Parameters:
			T_max: maximum number of iterations
			t_max: Value of n in the n step algorithm, here it is not taken into account
			gamma: depreciation of the futur
			learning_rate: learning_rate of the optimiser
			Iasyncupdate: Number of steps between two updates
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
		super(slave_worker_1_step_sarsa, self).__init__(**kwargs)
		self.T_max = T_max
		self.gamma = gamma
		self.env = gym.make(env_name)
		self.output_size = self.env.action_space.n
		self.input_size = self.env.observation_space.shape[0]
		self.verbose = verbose
		self.epsilon_ini = epsilon_ini
		self.weighted = weighted
		self.Iasyncupdate=Iasyncupdate

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

		self.qnn.initialisation()

		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		self.qnn.read_value_from_theta(self.sess)
		self.qnn.read_value_from_theta_minus(self.sess)

		epsilon = self.epsilon_ini
		t = 0
		y_batch = []
		nb_env = 0
		firstiter=True

		observation = self.env.reset()

		while settings.T.value<self.T_max:

			t_init = t
			

			if self.verbose:
				self.env.render()
				if settings.T.value%5000 == 0:
					print("T = %s"%settings.T.value)

			self.qnn.read_value_from_theta(self.sess)

			action = epsilon_greedy_policy(self.qnn, observation, epsilon, self.env, 
											self.sess, self.policy, self.weighted)

			observationprime, reward, done, info = self.env.step(action) 

			if done:
				y = reward
				observationprime = self.env.reset()
				t_init = t + 1
				nb_env += 1
			else:
				feed_dic = {self.qnn.variables["input_observation"]: observationprime.reshape((1, -1))}
				values = np.squeeze(self.sess.run(self.qnn.variables["y"], feed_dict=feed_dic))
				actionprime = epsilon_greedy_policy(self.qnn, observationprime, epsilon, self.env, 
											self.sess, self.policy, self.weighted)
				y = reward + self.gamma * values[actionprime]
			
			if firstiter:
				firstiter=False
				observation_batch = observation.reshape((1, -1))
				action_batch = [action]
			else:
				observation_batch = np.vstack((observation_batch, observation.reshape((1, -1))))
				action_batch.append(action)
			
			y_batch.append(y)
			observation = observationprime
			with settings.T.get_lock():
				settings.T.value += 1
			
			t += 1

			if epsilon>0.01:
				epsilon -= (self.epsilon_ini - 0.01)/25000

			if t %self.Iasyncupdate == 0:
			   
				action_batch_multiplier = np.eye(self.output_size)[action_batch].T
				
				y_batch_arr = np.array(y_batch).reshape((-1, 1))

				shuffle = range(len(y_batch_arr))
				np.random.shuffle(shuffle)

				self.qnn.read_value_from_theta(self.sess)

				feed_dict = {self.qnn.variables["input_observation"]: observation_batch[shuffle, :],
							 self.qnn.variables["y_true"]: y_batch_arr[shuffle, :], 
							 self.qnn.variables["y_action"]: action_batch_multiplier[:, shuffle]}
				self.sess.run(self.qnn.train_step, feed_dict=feed_dict)

				self.qnn.assign_value_to_theta(self.sess)

				firstiter = True
				y_batch = []

		return
