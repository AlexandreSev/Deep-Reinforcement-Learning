# coding: utf-8

import gym
import multiprocessing as mp
import numpy as np

from ..utils.utils import epsilon_greedy_policy
from ..utils import settings
from ..utils import callback as cb

from ..neuralnets import qnn

class slave_worker_n_step(mp.Process):
	"""
	The class coding a process to run a environment gym in itself and sharing gradient updates. 
	This slave uses asynchrone n step Q learning algorithm.
	"""

	def __init__(self, T_max=100000, t_max=5, gamma=0.9, learning_rate=0.001, Iasyncupdate=10,
				 env_name="CartPole-v0", model_option={"n_hidden":1, "hidden_size":[10]}, 
				 verbose=False, policy=None, epsilon_ini=0.9, alpha_reg=0., beta_reg=0.01, 
				 weighted=False, eps_fall=50000, callback=None, callback_name="callbacks/actor0", 
				 callback_batch_size=100, **kwargs):
		"""
		Parameters:
			T_max: maximum number of iterations
			t_max: Value of n in the n step algorithm
			gamma: depreciation of the futur
			learning_rate: learning_rate of the optimiser
			Iasyncupdate: Not used here
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
		super(slave_worker_n_step, self).__init__(**kwargs)
		self.T_max = T_max
		self.t_max = t_max
		self.gamma = gamma
		self.env = gym.make(env_name)
		self.output_size = self.env.action_space.n
		self.input_size = self.env.observation_space.shape[0]
		self.verbose = verbose
		self.epsilon_ini = epsilon_ini
		self.weighted = weighted
		self.eps_fall = eps_fall

		if callback:
			self.callback = cb.callback(batch_size=callback_batch_size, saving_directory=callback_name, 
									observation_size=self.input_size)
		else:
			self.callback = None

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
		nb_env = 0
		rpe = 0
		t_env = 0

		observation = self.env.reset()

		rewards_env = []
		estimated_rewards_env = []

		while settings.T.value<self.T_max:

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
					if settings.T.value%5000 == 0:
						print("T = %s"%settings.T.value)

				self.qnn.read_value_from_theta(self.sess)

				random, action = epsilon_greedy_policy(self.qnn, observation, epsilon, self.env, 
												self.sess, self.policy, self.weighted)

				observation, reward, done, info = self.env.step(action) 

				t_env +=1
				if t_env >500:
					done = True
					t_env = 0

				if self.callback:
					rewards_env.append(reward)

					rpe += reward

				reward_batch.append(reward)
				action_batch.append(action)

				if self.callback:
					self.callback.store(reward, random, action, observation_batch[0])

				observation_batch = np.vstack((observation.reshape((1, -1)), observation_batch))
				#print("reward_batch", reward_batch)
				#print("action_batch", action_batch)
				#print("observation_batch", observation_batch)
			  

				if done:
					nb_env += 1
					observation = self.env.reset()
					if self.callback:
						self.callback.store_rpe(rpe)
					rpe = 0
				
				with settings.T.get_lock():
					settings.T.value += 1
				
				t += 1

				if epsilon>0.01:
					epsilon -= (self.epsilon_ini - 0.01)/self.eps_fall
			
			if done:
				R = 0
			else:
				R = self.qnn.best_reward(observation, self.sess, self.weighted)

			true_reward = []
			for i in range(t - 1, t_init - 1, -1):
				R = reward_batch[i] + self.gamma * R
				true_reward.append(R)

			if self.callback:
				estimated_rewards_env += true_reward[::-1]
				if done:
					history = np.zeros((len(estimated_rewards_env), 4))
					history[:, 0] = np.arange(len(estimated_rewards_env))
					history[:, 1] = rewards_env
					history[:, 2] = estimated_rewards_env
					history[:, 3] = nb_env * np.ones(len(estimated_rewards_env))
					self.callback.write_history(history)
					rewards_env = []
					estimated_rewards_env = []


			action_batch.reverse()
			action_batch_multiplier = np.eye(self.output_size)[action_batch].T
			y_batch_arr = np.array(true_reward).reshape((-1, 1))

			#print("fed_reward_batch", y_batch_arr)
			#print("fed_action_batch", action_batch_multiplier)
			#print("fed_observation_batch", observation_batch[:-1, :])

			shuffle = range(len(y_batch_arr))
			np.random.shuffle(shuffle)
			
			self.qnn.read_value_from_theta(self.sess)
			
			feed_dict = {self.qnn.variables["input_observation"]: observation_batch[1:, :][shuffle, :],
						 self.qnn.variables["y_true"]: y_batch_arr[shuffle, :], 
						 self.qnn.variables["y_action"]: action_batch_multiplier[:, shuffle]}
			self.sess.run(self.qnn.train_step, feed_dict=feed_dict)


			diff = self.qnn.assign_value_to_theta(self.sess)

			if self.callback:
				self.callback.store_diff(diff)

		return
