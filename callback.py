# coding: utf-8
import numpy as np
import os
from os.path import join as pjoin


class callback():

	def __init__(self, batch_size=100, saving_directory="./", observation_size=4):

		self.batch_size = batch_size
		self.counter = 0
		self.rewards = np.zeros(batch_size)
		self.random = np.zeros(batch_size)
		self.action = np.zeros(batch_size)
		self.observation = np.zeros((batch_size, observation_size))
		self.diff = []
		self.rpe = []

		self.saving_directory = saving_directory
		self.list_directory = ["rewards.csv", "random.csv", "action.csv", "observation.csv", 
								"diff.csv", "rpe.csv"]
		self.data = [self.rewards, self.random, self.action, self.observation, self.diff, self.rpe]
		self.init()


	def init(self):
		if os.path.exists(self.saving_directory):
			for f in os.listdir(self.saving_directory):
				os.remove(pjoin(self.saving_directory, f))
		else:
			os.makedirs(self.saving_directory)



	def store(self, reward, random, action, observation):
		self.rewards[self.counter] = reward
		self.random[self.counter] = random
		self.action[self.counter] = action
		self.observation[self.counter] = observation
		self.counter += 1

		if self.counter == self.batch_size:
			self.write_on_disk()

	def store_diff(self, diff):
		self.diff.append(diff)

	def store_rpe(self, rpe):
		self.rpe.append(rpe)

	def write_on_disk(self):
		self.data = [self.rewards, self.random, self.action, self.observation, self.diff, self.rpe]
		self.diff = np.array(self.diff)
		for data, name in zip(self.data, self.list_directory):
			with open(pjoin(self.saving_directory, name), 'ab') as writer:
				np.savetxt(writer, data, delimiter=";")
		self.counter = 0
		self.diff = []
		print(self.rpe)
		self.rpe = []

