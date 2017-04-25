# coding: utf-8
import numpy as np
from ..utils import settings
import os
from os.path import join as pjoin

def critere_keys(key):
    """
    Return True if the key corresponds to a weight or a bias
    Parameters:
        key: string
        minus: Boolean, If true, critere_keys returns True if the key corresponds 
               to a parameter of theta minus, else, it returns True for a parameter of theta
    """
    critere = (key not in ["input_observation", "y_true", "y_action", "actions", "values"])
    critere = critere & (key[-3:] != "_ph") & (key[-7:] != "_assign")

    return critere


class A3CNeuralNetwork():
    """
    Class coding a Neural Network which evaluates the reward. In input, it will take a state (variable 
        observations) and will return the reward estimated for each action.
    """

    def __init__(self, input_size=4, output_size=2, n_hidden=2, hidden_size=[128, 64], 
                learning_rate=0.001, alpha_reg=0., beta_reg=0.01):
        """
        Parameters:
            input_size: size of observations, only 1D array are accepted for now
            output_size: number of possible action
            n_hidden: number of hidden layers, must be stricly positiv
            hidden_size: list of size of the hidden layers
            learning_rate: learning_rate used in the RMSPROP
            alpha_reg: coefficient of l1 regularisation
            beta_reg: coefficient of l2 regularisation
        """

        self.input_size = input_size
        self.output_size = output_size
        self.n_hidden = n_hidden
        self.hidden_size = hidden_size

        assert (self.n_hidden == len(self.hidden_size)), "Ill defined hidden layers"
        assert (self.n_hidden > 0), "Not implemented yet"

        self.variables = {}
        self.initialised = False
        self.learning_rate = learning_rate
        self.alpha_reg = alpha_reg
        self.beta_reg = beta_reg

        self.theta_copy = []


    def initialisation(self):
        """
        Create the neural network, the loss and the train step
        """
        self.create_variables()
        self.create_placeholders()
        self.build_model()
        self.reset_lr(None, True)
        self.build_loss()
        self.build_train_step()
        self.initialised = True

    def create_summary(self, sess, name="0"):
        import tensorflow as tf

        save_path = './callbacks/summaries/' + name + '/'

        tf.summary.scalar("learning rate", self.decay_learning_rate)
        tf.summary.scalar("loss", self.loss_vf)

        keys = sorted(self.variables.keys())
        keys = [ key for key in keys if critere_keys(key)]

        for key in keys:
            tf.summary.histogram(key, self.variables[key])

        self.merged = tf.summary.merge_all()

        if os.path.exists(save_path):
            for f in os.listdir(save_path):
                os.remove(pjoin(save_path, f))
        else:
            os.makedirs(save_path)

        self.writer = tf.summary.FileWriter(save_path, sess.graph)

    def create_weight_variable(self, shape, name="W", type_layer=None):
        """
        Create a matrix of weights as a tf variable. Create also a placeholder and a assign operation
        to change it value.
        Parameters:
            shape: 2-uple, size of the matrix
            name: name under which the variable is saved
        """
        import tensorflow as tf

        self.variables[name] = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
        self.variables[name + "_assign_ph"] = tf.placeholder(tf.float32, shape=shape, 
            name=name+"_assign_ph")
        self.variables[name + "_assign"] = tf.assign(self.variables[name], 
            self.variables[name + "_assign_ph"])

    def create_bias_variable(self, shape, name="b", type_layer=None):
        """
        Create a biais as a tf variable. Create also a placeholder and a assign operation
        to change it value.
        Parameters:
            shape: 2-uple, size of the biais
            name: name under which the variable is saved
        """
        import tensorflow as tf

        self.variables[name] = tf.Variable(tf.constant(0.1, shape=shape), name=name)
        self.variables[name + "_assign_ph"] = tf.placeholder(tf.float32, shape=shape, 
            name=name+"_assign_ph")
        self.variables[name + "_assign"] = tf.assign(self.variables[name], 
            self.variables[name + "_assign_ph"])

    def create_variables(self):
        """
        Create all weight/biais variables for a full forward pass
        Parameters:
            name: string, used to complete every name of variables, usefull to create two NNs
        """
        self.create_weight_variable([self.input_size, self.hidden_size[0]], name="W1")
        self.create_bias_variable((1, self.hidden_size[0]), name="b1")

        for i in range(self.n_hidden-1):
            self.create_weight_variable([self.hidden_size[i], self.hidden_size[i+1]], 
                                        name="W"+str(i+2))
            self.create_bias_variable((1, self.hidden_size[i+1]), name="b"+str(i+2))


        self.create_weight_variable([self.hidden_size[-1], self.output_size],
            name="Wo_policy", type_layer="policy")
        self.create_bias_variable((1, self.output_size), name="bo_policy",
            type_layer="policy")

        self.create_weight_variable([self.hidden_size[-1], 1], name="Wo_vf",
            type_layer="vf")
        self.create_bias_variable((1, 1), name="bo_vf", type_layer="vf")

    def create_placeholders(self):
        """
        Create placeholders for the observations, the results and the actions took
        """
        import tensorflow as tf

        self.variables["input_observation"] = tf.placeholder(tf.float32, 
            shape=[None, self.input_size], name="i_observation")

        self.variables["y_true"] = tf.placeholder(tf.float32, shape=[None], name="y_true")

        self.variables["y_action"] = tf.placeholder(tf.float32, shape=[None, self.output_size], 
            name="action")

        self.variables["loss_policy_ph"] = tf.placeholder(tf.float32, shape=[None], 
            name="loss_policy")

    def build_model(self):
        """
        Create the forward pass
        Parameters:
            name:String, name used in create_variables
        """
        import tensorflow as tf
        
        y = tf.nn.relu(tf.matmul(self.variables["input_observation"], self.variables["W1"]) + 
                       self.variables["b1"], name="y1")
        
        for i in range(self.n_hidden-1):
            y = tf.nn.relu(tf.matmul(y, self.variables["W"+str(i+2)]) + 
                           self.variables["b"+str(i+2)], name="y"+str(i+2))
        
        self.variables["actions"] = tf.nn.softmax(tf.matmul(y, self.variables["Wo_policy"]) + self.variables["bo_policy"])
        self.variables["values"] = tf.matmul(y, self.variables["Wo_vf"]) + self.variables["bo_vf"]
        
    def build_loss(self):
        import tensorflow as tf

        log_pi = tf.log(tf.clip_by_value(self.variables["actions"], 1e-10, 1.))
        pi_actions = tf.reduce_sum(tf.multiply(log_pi , self.variables["y_action"]), axis=1)
        

        self.loss_policy = - tf.reduce_sum(tf.multiply(pi_actions, self.variables["loss_policy_ph"])) \
            - self.beta_reg * tf.reduce_sum(tf.multiply(self.variables["actions"], log_pi))

        self.loss_vf = tf.nn.l2_loss(self.variables["values"] - self.variables["y_true"])
        

    def build_train_step(self):
        import tensorflow as tf

        self.optimizer = tf.train.RMSPropOptimizer(self.decay_learning_rate,
            decay=0.99, momentum=0., centered=True)

        keys = self.variables.keys()
        keys = [key for key in keys if (key not in ["input_observation", "y_true", "y_action", "actions", "values"]) & (key[-3:] != "_ph") & \
                (key[-7:] != "_assign")]

        self.updates = self.optimizer.compute_gradients(self.loss_policy, 
                                [self.variables[key] for key in keys])
        self.updates += self.optimizer.compute_gradients(self.loss_vf, 
                                [self.variables[key] for key in keys])
        
        self.train_step = self.optimizer.apply_gradients(self.updates,
            global_step=self.global_step)

    def reset_lr(self, sess, init=False):
        import tensorflow as tf
        
        if init:
            self.global_step = tf.Variable(0, trainable=False)
            self.global_step_pl = tf.placeholder(tf.int32)
            self.global_step_assign = tf.assign(self.global_step, self.global_step_pl)

            self.decay_steps = tf.Variable(1, trainable=False)
            self.decay_steps_pl = tf.placeholder(tf.int32)
            self.decay_steps_assign = tf.assign(self.decay_steps, self.decay_steps_pl)
        else:
            print("Reset")
            temp = sess.run(self.decay_steps) * 2
            sess.run(self.decay_steps_assign, feed_dict={self.decay_steps_pl: temp})
        
        #self.decay_learning_rate = tf.train.exponential_decay(self.learning_rate,
        #    global_step=self.global_step, decay_steps=self.decay_steps, decay_rate=0.999)

        self.decay_learning_rate = self.learning_rate - self.global_step / self.decay_steps * self.learning_rate * 1e-3

    def get_reward(self, observation, sess):
        feed_dic = {self.variables["input_observation"]: observation.reshape((1, -1))}
        reward = np.squeeze(sess.run(self.variables["actions"], feed_dict=feed_dic))
        return reward

    def best_choice(self, observation, sess):
        """
        Return the best action and the estimated reward for a given state
        Parameters:
            observation: np.array, state of the environnement
            sess: tensorflow session, allow multiprocessing
        """
        assert self.initialised, "This model must be initialised (self.initialisation())"
        feed_dic = {self.variables["input_observation"]: observation.reshape((1, -1))}
        reward = np.squeeze(sess.run(self.variables["actions"], feed_dict=feed_dic))
        reward[-1] = 1 - np.sum(reward[:-1])
        value = sess.run(self.variables["values"], feed_dict = feed_dic)
      
        return np.random.choice(range(len(reward)), p = reward), value[0, 0]

    def weighted_choice(self, observation, sess):
        """
        Return a random action weighted by estimated reward
        Parameters:
            observation: np.array, state of the environnement
            sess: tensorflow session, allow multiprocessing
        """
        assert(self.initialised, "This model must be initialised (self.initialisation())")
        reward = self.get_reward(observation, sess)

        proba = [i/np.sum(reward) for i in reward]
        proba[-1] = 1 - np.sum(proba[:-1])
        action = np.random.choice(range(len(reward)), p=proba)
      
        return action, reward[action]

    def best_action(self, observation, sess, weighted=False):
        """
        Return the best action for a given state
        Parameters:
            observation: np.array, state of the environnement
            sess: tensorflow session, allow multiprocessing
            weighted: If True, return a random aciton weighted with estimated reward. 
                      Else, the best one.
        """
        if weighted:
            return self.weighted_choice(observation, sess)[0]
        else:
            return self.best_choice(observation, sess)[0]

    def best_reward(self, observation, sess, weighted=False):
        """
        Return the estimated reward for a given state
        Parameters:
            observation: np.array, state of the environnement
            sess: tensorflow session, allow multiprocessing
            weighted: If True, return a random aciton weighted with estimated reward. 
                      Else, the best one.
        """
        if weighted:
            return self.weighted_choice(observation, sess)[1]
        else:
            return self.best_choice(observation, sess)[1]

    def assign_value_to_theta(self, sess):
        """
        Assign the value of the NN weights to theta
        Parameters: 
            sess: tensorflow session, allow multiprocessing
        """
        import tensorflow as tf

        assert(self.initialised, "This model must be initialised (self.initialisation()).")

        keys = sorted(self.variables.keys())
        keys = [ key for key in keys if critere_keys(key)]

        diff = 0
        for i, key in enumerate(keys):
            diff_temp = sess.run(self.variables[key]) - self.theta_copy[i]
            settings.l_theta[i] += diff_temp
            diff += np.linalg.norm(diff_temp)
        return diff
        
    def assign_value_to_theta_prime(self, theta_prime):
        """
        Assign the value of theta to theta'
        Parameters: 
            sess: tensorflow session, allow multiprocessing
        """
        assert self.initialised, "This model must be initialised (self.initialisation())."

        keys = sorted(self.variables.keys())
        keys = [ key for key in keys if critere_keys(key)]

        for i, key in enumerate(keys):
            theta_prime[i] = settings.l_theta[i]

        return theta_prime
        
    def read_value_from_theta(self, sess, theta):
        """
        Assign the value of theta to the weights of the NN
        Parameters: 
            sess: tensorflow session, allow multiprocessing
        """
        import tensorflow as tf

        assert self.initialised, "This model must be initialised (self.initialisation())."

        self.theta_copy = []
        keys = sorted(self.variables.keys())
        keys = [ key for key in keys if critere_keys(key)]

        for i, key in enumerate(keys):
            self.theta_copy.append(theta[i])
            feed_dict = {self.variables[key + "_assign_ph"]: theta[i]}
            sess.run(self.variables[key + "_assign"], feed_dict=feed_dict)
