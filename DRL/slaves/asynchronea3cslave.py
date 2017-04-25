# coding: utf-8

import gym
import multiprocessing as mp
import numpy as np

from ..utils.utils import epsilon_greedy_policy, initialise_a3c
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
                 callback_batch_size=100, name="", seed=42, action_replay=1,  reset=False, **kwargs):
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

        if type(self.env.observation_space) == gym.spaces.discrete.Discrete:
            self.input_size = [self.env.observation_space.n]
            self.decode_obs = lambda r: np.eye(self.env.observation_space.n)[r]
        else:
            self.input_size = [self.env.observation_space.shape[0]]
            self.decode_obs = lambda r: r

        if type(self.env.action_space) == gym.spaces.tuple_space.Tuple:
            self.output_size = []
            for space in self.env.action_space.spaces:
                if type(space) == gym.spaces.discrete.Discrete:
                    self.output_size.append(space.n)
                else:
                    NotImplementedError
        else:
            self.output_size = [self.env.action_space.n]

        self.verbose = verbose
        self.epsilon_ini = epsilon_ini
        self.weighted = weighted
        self.eps_fall = eps_fall
        self.name = name
        self.seed = seed
        self.action_replay = action_replay
        self.reset = reset
        self.count_T_reset = 0
        self.lr_ini = learning_rate

        if callback:
            self.callback = cb.callback(batch_size=callback_batch_size, saving_directory=callback_name, 
                                    observation_size=self.input_size, action_size=len(self.output_size))
        else:
            self.callback = None

        if policy is None:
            self.policy = self.env.action_space.sample
        else:
            self.policy = policy

        self.a3cnn = a3cnn.A3CNeuralNetwork(input_size=self.input_size, output_size=self.output_size, 
                n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"], 
                learning_rate=learning_rate, alpha_reg=alpha_reg, beta_reg=beta_reg)

        self.theta_prime = initialise_a3c(n_hidden=model_option["n_hidden"],
                                          hidden_size=model_option["hidden_size"],
                                          input_size=self.input_size,
                                          output_size=self.output_size)
            
    def run(self):
        """
        Run the worker and launch the n step algorithm
        """
        import tensorflow as tf

        np.random.seed(self.seed)

        self.a3cnn.initialisation()

        self.sess = tf.Session()

        self.a3cnn.create_summary(self.sess, self.name)
        
        self.sess.run(tf.global_variables_initializer())

        epsilon = 1
        nb_env = 0
        rpe = 0
        t_env = 0
        action_replay = 1

        observation = self.env.reset()
        observation = self.decode_obs(observation)

        rewards_env = []
        estimated_rewards_env = []
        rewards = []

        minlr = self.lr_ini / 10

        while settings.T.value<self.T_max:

            t = 0
            done = False

            observation_batch = observation.reshape((1, -1))

            reward_batch = []
            action_batch = []

            self.theta_prime = self.a3cnn.assign_value_to_theta_prime(self.theta_prime)
            self.a3cnn.read_value_from_theta(self.sess, self.theta_prime)
            
            while (not done) & (t<=self.t_max):
                if self.verbose:
                    self.env.render()
                    if settings.T.value%5000 == 0:
                        print("T = %s"%settings.T.value)

                self.sess.run(self.a3cnn.global_step_assign, 
                    feed_dict={self.a3cnn.global_step_pl: settings.T.value - self.count_T_reset})

                if action_replay == 1:
                    random, action = epsilon_greedy_policy(self.a3cnn, observation, epsilon, self.env, 
                                                    self.sess, self.policy, self.weighted)
                    action_replay = self.action_replay
                else:
                    action_replay -= 1

                observation, reward, done, info = self.env.step(action)
                observation = self.decode_obs(observation)

                t_env +=1
                
                if self.callback:
                    rewards_env.append(reward)
                    rewards.append(self.a3cnn.get_reward(observation, self.sess))
                    rpe += reward

                reward_batch.append(reward)
                action_batch.append(action)

                if self.callback:
                    self.callback.store(reward, random, action, observation_batch[0], rewards)

                observation_batch = np.vstack((observation.reshape((1, -1)), observation_batch))

                if done:
                    nb_env += 1
                    observation = self.env.reset()
                    observation = self.decode_obs(observation)
                    if self.callback:
                        self.callback.store_rpe(rpe)
                        self.callback.store_hp(epsilon, self.sess.run(self.a3cnn.decay_learning_rate))
                    rpe = 0
                
                with settings.T.get_lock():
                    settings.T.value += 1
                
                t += 1

                if epsilon > self.epsilon_ini:
                    epsilon -= (1 - self.epsilon_ini)/self.eps_fall

                if self.reset & (self.sess.run(self.a3cnn.decay_learning_rate) < minlr) :
                    minlr /= 2
                    epsilon = 1
                    self.count_T_reset = settings.T.value
                    self.a3cnn.reset_lr(self.sess)
            
            if done:
                R = 0
            else:
                R = self.sess.run(self.a3cnn.variables["values"],
                    feed_dict={self.a3cnn.variables["input_observation"]: observation.reshape((1, -1))})[0,0]

            observation_batch = observation_batch[1:, :]
            action_batch.reverse()
            if len(self.output_size) == 1:
                action_batch = [[i] for i in action_batch]
            action_batch = np.array(action_batch)

            action_batch_multiplier = [np.eye(self.output_size[0])[action_batch[:, 0]]]
            for i in range(1, len(self.output_size)):
                action_batch_multiplier.append(np.eye(self.output_size[i])[action_batch[:,i]])

            true_reward = []
            policy_loss = []
            for i in range(t - 1, -1, -1):
                R = reward_batch[i] + self.gamma * R
                vf = self.sess.run(self.a3cnn.variables["values"],
                    feed_dict={self.a3cnn.variables["input_observation"]: observation_batch[i].reshape((1, -1))})[0,0]
                true_reward.append(R)
                policy_loss.append(R - vf)

            if self.callback:
                estimated_rewards_env += true_reward[::-1]
                if done:
                    history = np.zeros((len(estimated_rewards_env), 4 + np.sum(self.output_size)))
                    history[:, 0] = np.arange(len(estimated_rewards_env))
                    history[:, 1] = rewards_env
                    history[:, 2] = estimated_rewards_env
                    history[:, 3] = nb_env * np.ones(len(estimated_rewards_env))
                    temp = np.array([np.concatenate(i) for i in rewards])
                    history[:, 4:] = temp
                    self.callback.write_history(history)
                    rewards_env = []
                    estimated_rewards_env = []
                    rewards = []

            y_batch_arr = np.array(true_reward)
            policy_loss = np.array(policy_loss)

            shuffle = np.arange(len(y_batch_arr))
            np.random.shuffle(shuffle)
            
            feed_dict = {self.a3cnn.variables["input_observation"]: observation_batch[shuffle, :],
                         self.a3cnn.variables["y_true"]: y_batch_arr[shuffle], 
                         self.a3cnn.variables["loss_policy_ph"]: policy_loss[shuffle]}
            for i in range(len(self.output_size)):
                feed_dict[self.a3cnn.variables["y_action"][i]] = action_batch_multiplier[i][shuffle, :]
            

            #print(self.sess.run(self.a3cnn.updates, feed_dict=feed_dict))
            summary, _ = self.sess.run([self.a3cnn.merged, self.a3cnn.train_step], feed_dict=feed_dict)

            self.a3cnn.writer.add_summary(summary, t_env)

            diff = self.a3cnn.assign_value_to_theta(self.sess)

            if self.callback:
                self.callback.store_diff(diff)

        return
