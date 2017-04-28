# coding: utf-8
import numpy as np
import multiprocessing as mp
import time
import gym
import os
from os.path import join as pjoin

from ..neuralnets import a3cnn, qnn 
from ..utils.utils import epsilon_greedy_policy

from ..utils import settings
from ..utils import callback as cb

class tester_worker(mp.Process):
    """
    Worker wich will test if the environment is solved. It will also update theta minus.
    """
    
    def __init__(self, algo="nstep", T_max=100000, t_max=500, env_name="CartPole-v0", 
                model_option={"n_hidden":1, "hidden_size":[10]}, n_sec_print=10, 
                goal=495, len_history=100, Itarget=100, render=False, weighted=False,
                callback=None, callback_name="callbacks/tester", callback_batch_size=10, 
                checkpoint=600, checkpoints_path="./checkpoints", warmstart=False, 
                weights_path="./checkpoints/cartpole_v1_150/intermediate_weights", nb_render=1, 
                **kwargs):
        """
        Parameters:
            T_max: maximum number of iterations
            t_max: maximum number of timesteps per environment
            env_name: name of gym environnment
            model_option: dictionary, must have two keys. n_hidden defines the number of hidden layers,
                        hidden_size the size of them in the QNeuralNetwork used to estimate the reward
            n_sec_print: Number of seconds between two prints
            goal: Value of reward to considered the game as solved
            len_history: number of episodes to test the algorithm
            Itarget: number of iterations between two updates of theta minus
            render: If True, environment will be rendered
            kwargs: args of multiprocessing.Process
        """
        super(tester_worker, self).__init__(**kwargs)
        self.T_max = T_max
        self.t_max = t_max
        self.env = gym.make(env_name)

        if type(self.env.observation_space) == gym.spaces.discrete.Discrete:
            self.input_size = [self.env.observation_space.n]
            self.decode_obs = lambda r: np.eye(self.env.observation_space.n)[r]
        else:
            self.input_size = [self.env.observation_space.shape[0]]
            self.decode_obs = lambda r: r

        if type(self.env.action_space) == gym.spaces.box.Box:
            self.output_size = [50]
            action_range = self.env.action_space.high - self.env.action_space.low
            self.action_space = [self.env.action_space.low + action_range * (i+.5) / self.output_size 
                for i in range(self.output_size[0])]
        elif type(self.env.action_space) == gym.spaces.discrete.Discrete:
            self.output_size = [self.env.action_space.n]
            self.action_space = [i for i in range(self.output_size[0])]
        elif type(self.env.action_space) == gym.spaces.tuple_space.Tuple:
            self.output_size = []
            for space in self.env.action_space.spaces:
                if type(space) == gym.spaces.discrete.Discrete:
                    self.output_size.append(space.n)
                else:
                    NotImplementedError

        self.nb_env = 0
        self.Itarget = Itarget
        self.counter_T = Itarget
        self.render=render
        self.nb_render = nb_render
        self.weighted=weighted
        self.algo = algo
        
        if algo=="a3c":
            self.nn = a3cnn.A3CNeuralNetwork(input_size=self.input_size, output_size=self.output_size, 
                    n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
        else:
            self.nn = qnn.QNeuralNetwork(input_size=self.input_size, output_size=self.output_size, 
                    n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])

        self.history = [-1000 for i in range(len_history)]
        self.goal = goal
        self.max_mean = -1000
        self.current_mean = 0
        self.last_T = 0
        self.n_sec_print = n_sec_print
        self.policy=None

        if callback:
            self.callback = cb.callback(batch_size=callback_batch_size, saving_directory=callback_name, 
                                        observation_size=self.input_size, action_size=len(self.output_size))
        else:
            self.callback = None

        n_temp =  "Try_" + str(len(os.listdir(checkpoints_path)) )
        self.checkpoints_path = pjoin(checkpoints_path, n_temp)
        os.mkdir(self.checkpoints_path)
        self.final_weights_path = pjoin(self.checkpoints_path, "final_weights")
        self.best_weights_path = pjoin(self.checkpoints_path, "best_weights")
        self.checkpoints_path = pjoin(self.checkpoints_path, "intermediate_weights")
        self.checkpoint = checkpoint
        self.last_checkpoint = time.time()

        self.warmstart = warmstart
        self.weights_path = weights_path

    def add_history(self, reward):
        """
        Add a value to the history and remove the last one
        Parameters:
            reward: value to put in history
        """
        self.history = self.history[1:]
        self.history.append(reward)
        
    def stoping_criteria(self):
        """
        Check from the history if the game is solved 
        """
        self.current_mean = np.mean(self.history)
        if self.current_mean > self.max_mean:
            self.max_mean = self.current_mean
            self.saver.save(self.sess, self.best_weights_path)
        if (np.mean(self.history)<self.goal) :
            return False
        else:
            return True

    def run(self):
        """
        Launch the worker
        """
        import tensorflow as tf
        t_taken = time.time()

        self.nn.initialisation()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        if self.warmstart:
            self.nn.read_value_from_theta(self.sess, settings.l_theta)
            self.saver.restore(self.sess, self.weights_path)
            self.nn.assign_value_to_theta(self.sess)
            print("Model successfully loaded")

        observation = self.env.reset()
        observation = self.decode_obs(observation)

        epsilon=0.
        nb_env = 0

        t_init = time.time()
        while (settings.T.value<self.T_max) & (not self.stoping_criteria()):
            
            if time.time() - t_init > self.n_sec_print:
                print("T = %s"%settings.T.value)
                print("Max mean = %s"%self.max_mean)
                print("Current mean = %s"%self.current_mean)
                t_init = time.time()
            
            if time.time() - self.last_checkpoint > self.checkpoint:
                self.last_checkpoint = time.time()
                save_path = self.saver.save(self.sess, self.checkpoints_path)
                print("######################################")
                print("# Model saved in %s #"%save_path)
                print("######################################")



            t = 0
            self.nb_env += 1
            current_reward = 0

            self.nn.read_value_from_theta(self.sess, settings.l_theta)
            while t<self.t_max:

                if self.render & (nb_env % self.nb_render == 0):
                    self.env.render()

                t += 1

                if self.algo != "a3c" and settings.T.value >= self.counter_T:
                    self.counter_T += self.Itarget
                    for i, theta_minus in enumerate(settings.l_theta_minus):
                        settings.l_theta_minus[i] = settings.l_theta[i]

                _, action = epsilon_greedy_policy(self.nn, observation, epsilon, self.output_size,
                                                  self.sess, self.policy, self.weighted)
                
                if type(self.env.action_space) == gym.spaces.discrete.Discrete:
                    env_action = action[0]
                    action = env_action
                elif type(self.env.action_space) == gym.spaces.box.Box:
                    env_action = [self.action_space[action[i]] for i in range(len(action))]
                    action = action[0]
                else:
                    env_action = action

                observation_prime, reward, done, info = self.env.step(env_action)
                observation_prime = self.decode_obs(observation_prime)

                current_reward += reward

                if self.callback:
                    self.callback.store(reward, 0, action, observation, 0)

                if done:
                    #print("DONE in %s timesteps"%t)
                    observation = self.env.reset()
                    observation = self.decode_obs(observation)
                    self.add_history(current_reward)
                    t += self.t_max
                    if self.callback:
                        self.callback.store_rpe(current_reward)
                else:
                    observation = observation_prime

            if not done:
                observation = self.env.reset()
                observation = self.decode_obs(observation)
                self.add_history(current_reward)
                if self.callback:
                    self.callback.store_rpe(current_reward)
            else:
                self.last_T = settings.T.value

            nb_env += 1

        print("Training completed")
        save_path = self.saver.save(self.sess, self.final_weights_path)
        print("Model saved in %s"%save_path)
        print("T final = %s"%self.last_T)
        print("Done in %s environments"%(self.nb_env-100))
        print("Done in %s seconds"%(time.time() - t_taken))
        settings.T.value += self.T_max

        observation = self.env.reset()

        for i in range(10):
            t = 0
            done = False
            while t<self.t_max:
                t += 1
                self.env.render()

                action = self.nn.best_action(observation, self.sess)

                observation, reward, done, info = self.env.step(action) 

                if done:
                    print("Environment completed in %s timesteps"%t)
                    observation = self.env.reset()
                    t += self.t_max
            if not done:
                observation = self.env.reset()
                print("Environment last %s timesteps"%t)
        return