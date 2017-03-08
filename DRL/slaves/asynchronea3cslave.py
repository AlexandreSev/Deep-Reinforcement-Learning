# coding: utf-8

import gym
import multiprocessing as mp
import numpy as np

from ..utils.utils import epsilon_greedy_policy
from ..utils import settings
from ..utils import callback as cb

from ..neuralnets import a3cnn

class slave_worker_a3c(mp.Process):
    """
    The class coding a process to run a environment gym in itself and sharing gradient updates. 
    This slave uses asynchrone n step Q learning algorithm.
    """

    def __init__(self, T_max=100000, t_max=5, gamma=0.9, learning_rate=0.001, Iasyncupdate=10,
                 env_name="CartPole-v0", model_option={"n_hidden":1, "hidden_size":[10]}, 
                 verbose=False, policy=None, epsilon_ini=0.9, alpha_reg=0., beta_reg=0.001, 
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
        super(slave_worker_a3c, self).__init__(**kwargs)
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

        self.a3cnn = a3cnn.A3CNeuralNetwork(input_size=self.input_size, output_size=self.output_size, 
                n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"], 
                learning_rate=learning_rate, alpha_reg=alpha_reg, beta_reg=beta_reg)
            
        
    def run(self):
        """
        Run the worker and launch the n step algorithm
        """
        import tensorflow as tf

        self.a3cnn.initialisation()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.a3cnn.read_value_from_theta(self.sess)

        epsilon = self.epsilon_ini
        nb_env = 0
        rpe = 0

        observation = self.env.reset()

        while settings.T.value<self.T_max:

            t = 0
            t_init = t
            done = False

            observation_batch = observation.reshape((1, -1))

            reward_batch = []
            action_batch = []

            d_theta = {}
            keys = self.a3cnn.variables.keys()
            keys = [key for key in keys if (key not in ["input_observation", "y_true", "y_action", "actions", "values"]) & (key[-3:] != "_ph") & \
                    (key[-7:] != "_assign")]
            common_keys = [key for key in keys if ("policy" not in key) & ("vf" not in key)]
            policy_keys = [key for key in keys if "policy" in key]
            vf_keys = [key for key in keys if "vf" in key]
            for key in common_keys:
                d_theta[key + "_grad_policy_ph"] = 0
                d_theta[key + "_grad_vf_ph"] = 0
            for key in policy_keys:
                d_theta[key + "_grad_policy_ph"] = 0
            for key in vf_keys:
                d_theta[key + "_grad_vf_ph"] = 0

            self.a3cnn.read_value_from_theta(self.sess)


            while (not done) & (t-t_init<=self.t_max):
            
                if self.verbose:
                    self.env.render()
                    if settings.T.value%5000 == 0:
                        print("T = %s"%settings.T.value)

                random, action = epsilon_greedy_policy(self.a3cnn, observation, epsilon, self.env, 
                                                       self.sess, self.policy, self.weighted)

                observation, reward, done, info = self.env.step(action) 

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
                R = self.sess.run(self.a3cnn.variables["values"],
                    feed_dict={self.a3cnn.variables["input_observation"]: observation.reshape((1, -1))})[0,0]

            action_batch.reverse()
            for i in range(t - 1, t_init - 1, -1):
                R = reward_batch[i] + self.gamma * R
                feed_dict = {self.a3cnn.variables["input_observation"]: observation_batch[i+1,:].reshape((1, -1)),
                             self.a3cnn.variables["y_true"]: np.array([[R]]), 
                             self.a3cnn.variables["y_action"]: np.eye(2)[[action_batch[i]]].T}
                for key in d_theta.keys():
                    d_theta[key] += self.sess.run(self.a3cnn.gradients[key], feed_dict=feed_dict)

            feed_dict = {self.a3cnn.variables[key]: d_theta[key] for key in d_theta.keys()}
            #print(d_theta["W1_grad_policy_ph"])
            self.a3cnn.read_value_from_theta(self.sess)
            self.sess.run(self.a3cnn.train_step, feed_dict=feed_dict)

            diff = self.a3cnn.assign_value_to_theta(self.sess)

            if self.callback:
                self.callback.store_diff(diff)

        return
