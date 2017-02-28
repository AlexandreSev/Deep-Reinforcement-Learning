
# coding: utf-8

# In[1]:

import gym
import time

import numpy as np
np.random.seed(2)

import multiprocessing as mp
import ctypes


def weight_variable(shape, name=None):
    import tensorflow as tf
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    import tensorflow as tf
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def create_variable(name="", input_size=4, output_size=2, n_hidden=2, hidden_size=[128, 64]):
    import tensorflow as tf
    
    variables_dict = {}
    
    variables_dict["W1" + name] = weight_variable([input_size, hidden_size[0]], name="W1" + name)
    variables_dict["W1" + name + "_ph"] = tf.placeholder(tf.float32, shape=[input_size, hidden_size[0]], 
        name="W1"+name+"_ph")
    variables_dict["W1" + name + "_assign"] = tf.assign(variables_dict["W1" + name], 
        variables_dict["W1" + name + "_ph"])
    variables_dict["W1_grad" + name] = weight_variable([input_size, hidden_size[0]], name="W1_grad" + name)
    
    variables_dict["b1" + name] = bias_variable((1, hidden_size[0]), name="b1" + name)
    variables_dict["b1" + name + "_ph"] = tf.placeholder(tf.float32, shape=[1, hidden_size[0]], 
        name="b1"+name+"_ph")
    variables_dict["b1" + name + "_assign"] = tf.assign(variables_dict["b1" + name], 
        variables_dict["b1" + name + "_ph"])
    variables_dict["b1_grad" + name] = bias_variable((1, hidden_size[0]), name="b1_grad" + name)

    for i in range(n_hidden-1):
        variables_dict["W"+str(i+2) + name] = weight_variable([hidden_size[i], hidden_size[i+1]], 
            name="W"+str(i+2) + name)
        variables_dict["W"+str(i+2) + name + "_ph"] = tf.placeholder(tf.float32, 
            shape=[hidden_size[i], hidden_size[i+1]], name="W"+str(i+2)+name+"_ph")
        variables_dict["W"+str(i+2) + name + "_assign"] = tf.assign(variables_dict["W"+str(i+2) + name], 
            variables_dict["W"+str(i+2) + name + "_ph"])
        variables_dict["W"+str(i+2) + "_grad" + name] = weight_variable([hidden_size[i], hidden_size[i+1]], 
            name="W"+str(i+2) + "_grad" + name)
        

        variables_dict["b"+str(i+2) + name] = bias_variable((1, hidden_size[i+1]), name="b"+str(i+2) + name)
        variables_dict["b"+str(i+2) + name + "_ph"] = tf.placeholder(tf.float32, 
            shape=[1, hidden_size[i+1]], name="b"+str(i+2)+name+"_ph")
        variables_dict["b"+str(i+2) + name + "_assign"] = tf.assign(variables_dict["b"+str(i+2) + name], 
            variables_dict["b"+str(i+2) + name + "_ph"])
        variables_dict["b"+str(i+2) + "_grad" + name] = bias_variable((1, hidden_size[i+1]), name="b"+str(i+2) + "_grad" + name)

    variables_dict["Wo_policy" + name] = weight_variable([hidden_size[-1], output_size], name="Wo_policy" + name)
    variables_dict["Wo_policy" + name + "_ph"] = tf.placeholder(tf.float32, shape=[hidden_size[-1], output_size], 
        name="Wo_policy"+name+"_ph")
    variables_dict["Wo_policy" + name + "_assign"] = tf.assign(variables_dict["Wo_policy" + name], 
        variables_dict["Wo_policy" + name + "_ph"])
    variables_dict["Wo_policy_grad" + name] = weight_variable([hidden_size[-1], output_size], name="Wo_policy_grad" + name)

    variables_dict["bo_policy" + name] = bias_variable((1, output_size), name="bo_policy" + name)
    variables_dict["bo_policy" + name + "_ph"] = tf.placeholder(tf.float32, shape=[1, output_size], 
        name="bo_policy"+name+"_ph")
    variables_dict["bo_policy" + name + "_assign"] = tf.assign(variables_dict["bo_policy" + name], 
        variables_dict["bo_policy" + name + "_ph"])
    variables_dict["bo_policy_grad" + name] = bias_variable((1, output_size), name="bo_policy_grad" + name)

    variables_dict["Wo_vf" + name] = weight_variable([hidden_size[-1], 1], name="Wo_vf" + name)
    variables_dict["Wo_vf" + name + "_ph"] = tf.placeholder(tf.float32, shape=[hidden_size[-1], 1], 
        name="Wo_vf"+name+"_ph")
    variables_dict["Wo_vf" + name + "_assign"] = tf.assign(variables_dict["Wo_vf" + name], 
        variables_dict["Wo_vf" + name + "_ph"])
    variables_dict["Wo_vf_grad" + name] = weight_variable([hidden_size[-1], 1], name="Wo_vf_grad" + name)

    variables_dict["bo_vf" + name] = bias_variable((1, 1), name="bo_vf" + name)
    variables_dict["bo_vf" + name + "_ph"] = tf.placeholder(tf.float32, shape=[1, 1], 
        name="bo_vf"+name+"_ph")
    variables_dict["bo_vf" + name + "_assign"] = tf.assign(variables_dict["bo_vf" + name], 
        variables_dict["bo_vf" + name + "_ph"])
    variables_dict["bo_vf_grad" + name] = bias_variable((1, 1), name="bo_vf_grad" + name)

    variables_dict["input_observation"] = tf.placeholder(tf.float32, shape=[None, input_size], name="i_observation" + name)
    
    variables_dict["y_true"] = tf.placeholder(tf.float32, shape=[None, 1], name="y_true" + name)
    variables_dict["y_action"] = tf.placeholder(tf.float32, shape=[output_size, None], name="action" + name)
    
    return variables_dict

def build_model(variables_dict, name="", input_size=4, output_size=2, n_hidden=2, hidden_size=[128, 64]):
    """
    Create a simple model
    """
    import tensorflow as tf
    
    y = tf.nn.relu(tf.matmul(variables_dict["input_observation"], variables_dict["W1" + name]) + 
                   variables_dict["b1" + name], name="y1" + name)
    
    for i in range(n_hidden-1):
        y = tf.nn.relu(tf.matmul(y, variables_dict["W"+str(i+2) + name]) + 
                       variables_dict["b"+str(i+2) + name], name="y"+str(i+2) + name)
    
    values = tf.matmul(y, variables_dict["Wo_vf" + name]) + variables_dict["bo_vf" + name]
    actions = tf.nn.softmax(tf.matmul(y, variables_dict["Wo_vf" + name]) + variables_dict["bo_vf" + name])

    return actions, values

def build_loss(values, actions, variables_dict, alpha_reg=0, beta_reg=0.01):
    import tensorflow as tf
    loss_list_policy = tf.log(tf.matmul(actions, variables_dict["y_action"])) * (variables_dict["y_true"] - values)
    loss_policy = tf.reduce_mean(loss_list_policy)

    loss_list_values = tf.nn.l2_loss(values - variables_dict["y_true"])
    loss_values = tf.reduce_mean(loss_list_values)

    # l1_reg = 0
    # l2_reg = 0

    # keys = variables_dict.keys()
    # keys.sort()
    # keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "y"]) & (key[-3:] != "_ph") & \
    #         (key[-7:] != "_assign")]
    # for key in keys:
    #     l1_reg += tf.reduce_sum(tf.abs(variables_dict[key]))
    #     l2_reg += tf.nn.l2_loss(variables_dict[key])

    # loss += alpha_reg * l1_reg + beta_reg * l2_reg

    return loss_policy, loss_values

def compute_gradients(loss_policy, loss_values, variables_dict):
    import tensorflow as tf
    keys = variables_dict.keys()
    keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "y"]) & (key[-3:] != "_ph") & \
            (key[-7:] != "_assign") & (key[-5:] != "_grad")]
    grads = {}
    for key in keys:
        grads[key + "_policy"] = tf.gradients(loss_policy)
        grads[key + "_values"] = tf.gradients(loss_values)

    return grads

def build_train_step(variables_dict, grads):
    keys = variables_dict.keys()
    keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "y"]) & (key[-3:] != "_ph") & \
            (key[-7:] != "_assign") & (key[-5:] != "_grad")]
    updates = []
    for key in keys:
        updates.append((grads[key + "_values"], variables_dict[key]))
        updates.append((grads[key + "_policy"], variables_dict[key]))

    opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.5, centered=True)
    train_step = opt.apply_gradients(updates)
    return train_step

def best_choice(variables_dict, observation, sess):
    import tensorflow as tf
    feed_dic = {variables_dict["input_observation"]: observation.reshape((1, -1))}
    #print("Je passe")
    reward = sess.run(variables_dict["y"], feed_dict=feed_dic)
    
    """
    reward = [max(i, 0) for i in reward[0]]
    choice = np.random.uniform(low=0, high=np.sum(reward), size=1)
    action = 0
    count = reward[0]
    while (count <= choice) & (action < len(reward) - 1):
        action += 1
        count += max(0, reward[action])
    return action, reward[action]
    """

    return np.argmax(reward), np.max(reward)

def best_action(variables_dict, observation, sess):
    return best_choice(variables_dict, observation, sess)[0]

def best_reward(variables_dict, observation, sess):
    return best_choice(variables_dict, observation, sess)[1]

def epsilon_greedy_policy(variables_dict, observation, epsilon, env, sess, policy=None):
    u = np.random.binomial(1, epsilon)
    if u:
        if policy is None:
            return env.action_space.sample()
        else:
            action = policy()
            return action
    else:
        return best_action(variables_dict, observation, sess)


# In[5]:

def assign_value_to_theta(variables_dict, sess):
    global l_theta
    import tensorflow as tf
    keys = variables_dict.keys()
    keys.sort()
    keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "y"]) & (key[-3:] != "_ph") & \
            (key[-7:] != "_assign")]
    for i, key in enumerate(keys):
        l_theta[i] = sess.run(variables_dict[key])
    return l_theta
    
def read_value_from_theta(variables_dict, sess):
    global l_theta
    import tensorflow as tf
    keys = variables_dict.keys()
    keys.sort()
    keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "y"]) & (key[-3:] != "_ph") & \
            (key[-7:] != "_assign")]
    for i, key in enumerate(keys):
        feed_dict = {variables_dict[key + "_ph"]: l_theta[i]}
        sess.run(variables_dict[key + "_assign"], feed_dict=feed_dict)
    return variables_dict

def read_value_from_theta_minus(variables_dict, sess):
    global l_theta_minus
    import tensorflow as tf
    keys = variables_dict.keys()
    keys.sort()
    keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "y"]) & (key[-3:] != "_ph") & \
            (key[-7:] != "_assign")]
    for i, key in enumerate(keys):
        feed_dict = {variables_dict[key + "_ph"]: l_theta_minus[i]}
        sess.run(variables_dict[key + "_assign"], feed_dict=feed_dict)
    return variables_dict

def initialise(input_size=4, output_size=2, n_hidden=2, hidden_size=[128, 64]):
    l_theta = mp.Manager().list()
    
    shapes = [(input_size, hidden_size[0])]
    for i in range(n_hidden - 1):
        shapes.append((hidden_size[i], hidden_size[i+1]))
    shapes.append((hidden_size[-1], output_size))
    
    shapes.append((1, hidden_size[0]))
    for i in range(n_hidden - 1):
        shapes.append((1, hidden_size[i+1]))
    shapes.append((1, output_size))
    
    for i, shape in enumerate(shapes):
        l_theta.append(np.random.uniform(low=-0.01, high=0.01, size=shape))
        # l_theta[i].value = np.random.uniform(low=-0.01, high=0.01, size=shape)
        
    return l_theta


# In[6]:

class slave_worker(mp.Process):
    
    def __init__(self, T_max=100, Itarget=15, Iasyncupdate=10, gamma=0.9, learning_rate=0.001,
                   alpha_reg=0, beta_reg=0.01, env_name="CartPole-v0",
                   model_option={"n_hidden":1, "hidden_size":[10]}, 
                   verbose=False, policy=None, epsilon_ini=0.9, **kwargs):
        super(slave_worker, self).__init__(**kwargs)
        self.T_max = T_max
        self.Itarget = Itarget
        self.Iasyncupdate = Iasyncupdate
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.env = gym.make(env_name)
        self.verbose = verbose
        self.epsilon_ini = epsilon_ini

        if policy is None:
            self.policy = self.env.action_space.sample
        else:
            self.policy = policy
        
        self.variables_dict = create_variable(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
        self.variables_dict["values"], self.variables_dict["actions"] = build_model(self.variables_dict,
                                                                                    n_hidden=model_option["n_hidden"], 
                                                                                    hidden_size=model_option["hidden_size"])

        self.loss_policy, self.loss_values = build_loss(self.variables_dict["values"], self.variables_dict["actions"],
                                                        self.variables_dict, learning_rate=learning_rate,
                                                        alpha_reg=alpha_reg, beta_reg=beta_reg)
        self.train_step = build_train_step(self.variables_dict, "Un truc à méditer après le déjeuner")
        
        self.variables_dict_minus = create_variable(name="_minus", n_hidden=model_option["n_hidden"], 
                                                    hidden_size=model_option["hidden_size"])
        self.variables_dict_minus["values"], self.variables_dict_minus["actions"] = build_model(self.variables_dict_minus,
                                                                                                name="_minus",
                                                                                                n_hidden=model_option["n_hidden"], 
                                                                                                hidden_size=model_option["hidden_size"])
            
        
    def run(self):
        import tensorflow as tf
        global T, l_theta, l_theta_minus

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        epsilon = self.epsilon_ini
        t = 0
        y_batch = []
        nb_env = 0
        firstiter=True
        t_init = t

        observation = self.env.reset()

        while T.value<self.T_max:
            
            # if T.value %500 == 0:
            #     print(T.value)

            self.variables_dict = read_value_from_theta(self.variables_dict, self.sess)
            self.variables_dict_minus = read_value_from_theta(self.variables_dict_minus, self.sess)

            action = epsilon_greedy_policy(self.variables_dict, observation, epsilon, self.env, self.sess, self.policy)

            observationprime, reward, done, info = self.env.step(action) 

            if t - t_init > 200:
                done = True

            if done:
                y = reward
                observationprime = self.env.reset()
                t_init = t + 1
                nb_env += 1
            else:
                y = reward + self.gamma * best_reward(self.variables_dict_minus, observationprime, self.sess)
            
            if firstiter:
                firstiter=False
                observation_batch = observation.reshape((1, -1))
                action_batch = [action]
            else:
                observation_batch = np.vstack((observation_batch, observation.reshape((1, -1))))
                action_batch.append(action)
            
            y_batch.append(y)
            observation = observationprime
            with T.get_lock():
                T.value += 1
            
            t += 1
            if T.value %self.Itarget == 0:
                for i, theta_minus in enumerate(l_theta_minus):
                    l_theta_minus[i] = l_theta[i]
            
            if t %self.Iasyncupdate == 0:
               
                action_batch_multiplier = np.eye(2)[action_batch].T
                
                y_batch_arr = np.array(y_batch).reshape((-1, 1))

                self.variables_dict = read_value_from_theta(self.variables_dict, self.sess)
                
                feed_dict = {self.variables_dict["input_observation"]: observation_batch,
                             self.variables_dict["y_true"]: y_batch_arr, 
                             self.variables_dict["y_action"]: action_batch_multiplier}
                self.sess.run(self.train_step, feed_dict=feed_dict)

                l_theta = assign_value_to_theta(self.variables_dict, self.sess)

                firstiter = True
                y_batch = []

            if epsilon>0.01:
                epsilon -= (self.epsilon_ini - 0.01)/50000

        return


class master_worker(mp.Process):
    
    def __init__(self, T_max=1000, t_max=200, Itarget=15, nb_env=10, env_name="CartPole-v0", model_option={"n_hidden":1, "hidden_size":[10]}, 
                 verbose=False, **kwargs):
        import tensorflow as tf
        
        super(master_worker, self).__init__(**kwargs)
        self.T_max = T_max
        self.t_max = t_max
        self.env = gym.make(env_name)
        self.nb_env = nb_env
        self.Itarget = Itarget
        
        self.variables_dict = create_variable(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
        self.variables_dict["y"] = build_model(self.variables_dict, n_hidden=model_option["n_hidden"], 
                                               hidden_size=model_option["hidden_size"])

        self.history = [0 for i in range(200)]
        self.goal = 195
        self.max_mean = 0
        self.current_mean = 0
        self.last_T = 0

    def add_history(self, reward):
        self.history = self.history[1:]
        self.history.append(reward)
        
    def stoping_criteria(self):
        self.current_mean = np.mean(self.history)
        if self.current_mean > self.max_mean:
            self.max_mean = self.current_mean
        if (np.mean(self.history)<self.goal) :
            return False
        else:
            return True

    def run(self):
        global l_theta, l_theta_minus
        import tensorflow as tf

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        self.variables_dict = read_value_from_theta(self.variables_dict, self.sess)
        saver.save(self.sess, './ptb_rnnlm.weights')

        epsilon = 0.0
        observation = self.env.reset()

        t_init = time.time()
        while (T.value<self.T_max) & (not self.stoping_criteria()):
            #print(l_theta[-1])
            if time.time() - t_init > 10:
                print("T = %s"%T.value)
                print("Max mean = %s"%self.max_mean)
                print("Current mean = %s"%self.current_mean)
                t_init = time.time()

            t = 0
            while t<self.t_max:

                if T.value %self.Itarget == 0:
                    for i, theta_minus in enumerate(l_theta_minus):
                        l_theta_minus[i] = l_theta[i]
                t += 1
                #self.env.render()

                action = epsilon_greedy_policy(self.variables_dict, observation, epsilon, self.env, self.sess)

                observation, reward, done, info = self.env.step(action) 

                if done:
                    print("Environment completed in %s timesteps"%t)
                    observation = self.env.reset()
                    self.add_history(t)
                    t += self.T_max
            if not done:
                observation = self.env.reset()
                print("Environment last %s timesteps"%t)
                self.add_history(t)
                self.last_T = T.value
            else:
                self.variables_dict = read_value_from_theta(self.variables_dict, self.sess)

        print("Training completed")
        saver.save(self.sess, './end_training.weights')
        print("T final = %s"%self.last_T)
        T.value += self.T_max

        observation = self.env.reset()
        #self.variables_dict = read_value_from_theta(self.variables_dict, self.sess)

        for i in range(self.nb_env):
            t = 0
            done = False
            while t<self.t_max:
                t += 1
                self.env.render()

                action = best_action(self.variables_dict, observation, self.sess)

                observation, reward, done, info = self.env.step(action) 

                if done:
                    print("Environment completed in %s timesteps"%t)
                    observation = self.env.reset()
                    t += self.t_max
            if not done:
                observation = self.env.reset()
                print("Environment last %s timesteps"%t)
        return

def policy_template(x=0.5):
    return lambda :np.random.binomial(1, x)

def create_2D_policies(n):
    policies = []
    for i in range(n):
        policies.append(policy_template(1./(n+1) * (i+1)))
    return policies

def create_list_epsilon(n):
    e_list = [1, 0.5]
    p = [0.5, 0.5]
    return np.random.choice(e_list, n, p=p)

    e_max = 1
    e_min = 0.01
    return [e_min + i * (e_max-e_min) / n + (e_max-e_min) / (2*n) for i in range(n)]

def main(nb_process, T_max=5000,  model_option={"n_hidden":1, "hidden_size":[10]}, env_name="CartPole-v0"):
    global T, l_theta, l_theta_minus
    T = mp.Value('i', 0)
    
    l_theta = initialise(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
    l_theta_minus = initialise(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
    
    jobs = []
    policies = create_2D_policies(nb_process)
    epsilons = create_list_epsilon(nb_process)
    for i in range(nb_process):
        print("Process %s starting"%i)
        job = slave_worker(T_max=T_max, model_option=model_option, env_name=env_name, 
            policy=None, verbose=(i==15), epsilon_ini = epsilons[i])
        job.start()
        jobs.append(job)
    
    # for job in jobs:
    #    job.join()
    
    exemple = master_worker(T_max=T_max, t_max=200, model_option=model_option, env_name=env_name)
    exemple.start()
    exemple.join()
    
    """
    model.set_weights(theta.value)
    
    env = gym.make(env_name)
    observation = env.reset()

    for t in range(100000):
        env.render()
        #print(model.predict(np.append(observation, [0])), model.predict(np.append(observation, [1])))
        action = epsilon_greedy_policy(model, observation, 0.01)
        #action = weighted_choice(model, observation)[0]
        #print("NEXT")
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    """


# global T, theta, theta_minus
# model_option={"n_hidden":1, "hidden_size":[10]}
# env_name="CartPole-v0"
# T = mp.Value('i', 0)
# 
# model = build_model(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
# theta = mp.Array(ctypes.c_double, len(model.get_weights()))
# theta.value = model.get_weights()
# theta_minus  = mp.Array(ctypes.c_double, len(theta.value))
# theta_minus.value = theta.value
# 
# theta.value

# In[ ]:

if __name__=="__main__":
    import sys
    args = sys.argv
    if len(args)>2:
        main(int(args[1]), T_max=int(args[2]), model_option={"n_hidden":2, "hidden_size":[128, 128]})
    else:
        main(3, 50000)
