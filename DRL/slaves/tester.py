# coding: utf-8
import numpy as np
import multiprocessing as mp
import time
import gym

from ..neuralnets import a3cnn, qnn 
from ..utils.utils import epsilon_greedy_policy

from ..utils import settings

class tester_worker(mp.Process):
    """
    Worker wich will test if the environment is solved. It will also update theta minus.
    """
    
    def __init__(self, algo="nstep", T_max=100000, t_max=200, env_name="CartPole-v0", 
                model_option={"n_hidden":1, "hidden_size":[10]}, n_sec_print=10, 
                goal=195, len_history=100, Itarget=15, render=False, weighted=False, **kwargs):
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
        self.output_size = self.env.action_space.n
        self.input_size = self.env.observation_space.shape[0]
        self.nb_env = 0
        self.Itarget = Itarget
        self.counter_T = Itarget
        self.render=render
        self.weighted=weighted
        self.algo = algo
        
        if algo=="a3c":
            self.nn = ac3nn.A3CNeuralNetwork(input_size=self.input_size, output_size=self.output_size, 
                    n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
        else:
            self.nn = qnn.QNeuralNetwork(input_size=self.input_size, output_size=self.output_size, 
                    n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])

        self.history = [-1000 for i in range(len_history)]
        self.goal = goal
        self.max_mean = 0
        self.current_mean = 0
        self.last_T = 0
        self.n_sec_print = n_sec_print
        self.policy=None

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
        saver = tf.train.Saver()

        self.nn.read_value_from_theta(self.sess)
        saver.save(self.sess, './ptb_rnnlm.weights')

        observation = self.env.reset()

        epsilon=0.

        t_init = time.time()
        while (settings.T.value<self.T_max) & (not self.stoping_criteria()):
            
            if time.time() - t_init > self.n_sec_print:
                print("T = %s"%settings.T.value)
                print("Max mean = %s"%self.max_mean)
                print("Current mean = %s"%self.current_mean)
                t_init = time.time()

            t = 0
            self.nb_env += 1
            current_reward = 0
            while t<self.t_max:

                if self.render:
                    self.env.render()

                t += 1

                if self.algo != "a3c" and settings.T.value >= self.counter_T:
                    self.counter_T += self.Itarget
                    for i, theta_minus in enumerate(settings.l_theta_minus):
                        settings.l_theta_minus[i] = settings.l_theta[i]

                _, action = epsilon_greedy_policy(self.nn, observation, epsilon, self.env,
                                                  self.sess, self.policy, self.weighted)

                observation, reward, done, info = self.env.step(action) 

                current_reward += reward

                if done:
                    observation = self.env.reset()
                    self.add_history(current_reward)
                    t += self.T_max
            if not done:
                observation = self.env.reset()
                self.add_history(current_reward)
            else:
                self.last_T = settings.T.value
            self.nn.read_value_from_theta(self.sess)

        print("Training completed")
        saver.save(self.sess, './end_training.weights')
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