
# coding: utf-8

# In[1]:

import gym
import time

import numpy as np
np.random.seed(42)

import multiprocessing as mp
import ctypes


# In[2]:

def weight_variable(shape, name=None):
    import tensorflow as tf
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
    import tensorflow as tf
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


# In[3]:

def create_variable(name="", input_size=4, output_size=2, n_hidden=2, hidden_size=[128, 64]):
    import tensorflow as tf
    
    variables_dict = {}
    
    variables_dict["W1" + name] = weight_variable([input_size, hidden_size[0]], name="W1" + name)
    variables_dict["W1" + name + "_ph"] = tf.placeholder(tf.float32, shape=[input_size, hidden_size[0]], 
        name="W1"+name+"_ph")
    variables_dict["W1" + name + "_assign"] = tf.assign(variables_dict["W1" + name], 
        variables_dict["W1" + name + "_ph"])
    
    variables_dict["b1" + name] = bias_variable((1, hidden_size[0]), name="b1" + name)
    variables_dict["b1" + name + "_ph"] = tf.placeholder(tf.float32, shape=[1, hidden_size[0]], 
        name="b1"+name+"_ph")
    variables_dict["b1" + name + "_assign"] = tf.assign(variables_dict["b1" + name], 
        variables_dict["b1" + name + "_ph"])

    for i in range(n_hidden-1):
        variables_dict["W"+str(i+2) + name] = weight_variable([hidden_size[i], hidden_size[i+1]], 
            name="W"+str(i+2) + name)
        variables_dict["W"+str(i+2) + name + "_ph"] = tf.placeholder(tf.float32, 
            shape=[hidden_size[i], hidden_size[i+1]], name="W"+str(i+2)+name+"_ph")
        variables_dict["W"+str(i+2) + name + "_assign"] = tf.assign(variables_dict["W"+str(i+2) + name], 
            variables_dict["W"+str(i+2) + name + "_ph"])

        variables_dict["b"+str(i+2) + name] = bias_variable((1, hidden_size[i+1]), name="b"+str(i+2) + name)
        variables_dict["b"+str(i+2) + name + "_ph"] = tf.placeholder(tf.float32, 
            shape=[1, hidden_size[i+1]], name="b"+str(i+2)+name+"_ph")
        variables_dict["b"+str(i+2) + name + "_assign"] = tf.assign(variables_dict["b"+str(i+2) + name], 
            variables_dict["b"+str(i+2) + name + "_ph"])

    variables_dict["Wo" + name] = weight_variable([hidden_size[-1], output_size], name="Wo" + name)
    variables_dict["Wo" + name + "_ph"] = tf.placeholder(tf.float32, shape=[hidden_size[-1], output_size], 
        name="Wo"+name+"_ph")
    variables_dict["Wo" + name + "_assign"] = tf.assign(variables_dict["Wo" + name], 
        variables_dict["Wo" + name + "_ph"])

    variables_dict["bo" + name] = bias_variable((1, output_size), name="bo" + name)
    variables_dict["bo" + name + "_ph"] = tf.placeholder(tf.float32, shape=[1, output_size], 
        name="bo"+name+"_ph")
    variables_dict["bo" + name + "_assign"] = tf.assign(variables_dict["bo" + name], 
        variables_dict["bo" + name + "_ph"])

    variables_dict["input_observation"] = tf.placeholder(tf.float32, shape=[None, input_size], name="i_observation" + name)
    
    variables_dict["y_true"] = tf.placeholder(tf.float32, shape=[None, output_size], name="y_true" + name)
    variables_dict["y_action"] = tf.placeholder(tf.float32, shape=[None, output_size], name="action" + name)
    
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
    
    y = tf.matmul(y, variables_dict["Wo" + name]) + variables_dict["bo" + name]

    return y

def build_loss(y, variables_dict, learning_rate=0.001):
    import tensorflow as tf
    loss_list = tf.nn.l2_loss(y * variables_dict["y_action"] - variables_dict["y_true"])
    loss = tf.reduce_mean(loss_list)
    
    train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    return loss, train_step

def best_choice(variables_dict, observation, sess):
    import tensorflow as tf
    feed_dic = {variables_dict["input_observation"]: observation.reshape((1, -1))}
    #print("Je passe")
    reward = sess.run(variables_dict["y"], feed_dict=feed_dic)
    #print("Je casse")
    return np.argmax(reward), np.max(reward)

def best_action(variables_dict, observation, sess):
    return best_choice(variables_dict, observation, sess)[0]

def best_reward(variables_dict, observation, sess):
    return best_choice(variables_dict, observation, sess)[1]

def epsilon_greedy_policy(variables_dict, observation, epsilon, env, sess):
    u = np.random.binomial(1, epsilon)
    if u:
        return env.action_space.sample()
    else:
        return best_action(variables_dict, observation, sess)


# In[5]:

def assign_value_to_theta(l_theta, variables_dict, sess):
    import tensorflow as tf
    keys = variables_dict.keys()
    keys.sort()
    keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "y"]) & (key[-3:] != "_ph") & \
            (key[-7:] != "_assign")]
    for i, key in enumerate(keys):
        with l_theta[i].get_lock():
            l_theta[i].value = sess.run(variables_dict[key])
    return l_theta
    
def read_value_from_theta(l_theta, variables_dict, sess):
    import tensorflow as tf
    keys = variables_dict.keys()
    keys.sort()
    keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "y"]) & (key[-3:] != "_ph") & \
            (key[-7:] != "_assign")]
    for i, key in enumerate(keys):
        feed_dict = {variables_dict[key + "_ph"]: l_theta[i].value}
        sess.run(variables_dict[key + "_ph"], feed_dict=feed_dict)
    return variables_dict

def initialise(input_size=4, output_size=2, n_hidden=2, hidden_size=[128, 64]):
    l_theta = []
    
    shapes = [(input_size, hidden_size[0])]
    for i in range(n_hidden - 1):
        shapes.append((hidden_size[i], hidden_size[i+1]))
    shapes.append((hidden_size[-1], output_size))
    
    shapes.append((1, hidden_size[0]))
    for i in range(n_hidden - 1):
        shapes.append((1, hidden_size[i+1]))
    shapes.append((1, output_size))
    
    for i, shape in enumerate(shapes):
        l_theta.append(mp.Array(ctypes.c_double, shape))
        l_theta[i].value = np.random.uniform(low=-0.01, high=0.01, size=shape)
        
    return l_theta


# In[6]:

class slave_worker(mp.Process):
    
    def __init__(self, T_max=100, Itarget=15, Iasyncupdate=10, gamma=0.9, learning_rate=0.001, 
                   env_name="CartPole-v0", model_option={"n_hidden":1, "hidden_size":[10]}, verbose=False, **kwargs):
        super(slave_worker, self).__init__(**kwargs)
        self.T_max = T_max
        self.Itarget = Itarget
        self.Iasyncupdate = Iasyncupdate
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.env = gym.make(env_name)
        
        self.variables_dict = create_variable(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
        self.variables_dict["y"] = build_model(self.variables_dict, n_hidden=model_option["n_hidden"], 
                                               hidden_size=model_option["hidden_size"])

        self.loss, self.train_step = build_loss(self.variables_dict["y"], self.variables_dict)
        
        self.variables_dict_minus = create_variable(name="_minus", n_hidden=model_option["n_hidden"], 
                                                    hidden_size=model_option["hidden_size"])
        self.variables_dict_minus["y"] = build_model(self.variables_dict_minus, name="_minus", n_hidden=model_option["n_hidden"], 
                                                     hidden_size=model_option["hidden_size"])
        self.loss_minus, self.train_step_minus = build_loss(self.variables_dict_minus["y"], self.variables_dict_minus)
            
        
    def run(self):
        import tensorflow as tf
        global T, l_theta, l_theta_minus

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        self.variables_dict = read_value_from_theta(l_theta, self.variables_dict, self.sess)
        self.variables_dict_minus = read_value_from_theta(l_theta_minus, self.variables_dict_minus, self.sess)

        epsilon = 0.9
        t = 0
        x_batch = 0
        y_batch = []
        nb_env = 0
        firstiter=True
        observation = self.env.reset()
        t_init = t
        
        while T.value<self.T_max:
            
            if T.value %500 == 0:
                print(T.value)

            print(self.sess.run(self.variables_dict["bo"]))

            self.variables_dict = read_value_from_theta(l_theta, self.variables_dict, self.sess)

            self.variables_dict_minus = read_value_from_theta(l_theta_minus, self.variables_dict_minus, self.sess)

            action = epsilon_greedy_policy(self.variables_dict, observation, epsilon, self.env, self.sess)

            observationprime, reward, done, info = self.env.step(action) 

            if t - t_init > 200:
                done = True
            #print("Checkpoint 1")

            if done:
                y = reward
                observationprime = self.env.reset()
                t_init = t + 1
                nb_env += 1
            else:
                #print("Je passe")
                y = reward + self.gamma * best_reward(self.variables_dict_minus, observationprime, self.sess)
                #print("Je casse")
            
            #print("Checkpoint 2")
            
            if firstiter:
                firstiter=False
                observation_batch = observation.reshape((1, -1))
                action_batch = [action]
            else:
                observation_batch = np.vstack((observation_batch, observation.reshape((1, -1))))
                action_batch.append(action)
                
            #print("Checkpoint 3")
            
            y_batch.append(y)
            observation = observationprime
            with T.get_lock():
                T.value += 1
                
            #print("Checkpoint 4")
            
            t += 1
            if T.value %self.Itarget == 0:
                for i, theta_minus in enumerate(l_theta_minus):
                    with theta_minus.get_lock():
                        l_theta_minus[i].value = l_theta[i].value
                        
            #print("Checkpoint 5")
            
            if t %self.Iasyncupdate == 0:
                
                action_batch_arr = np.array(action_batch)
                action_batch_multiplier = np.eye(2)[action_batch_arr]
                
                sampling = np.random.randint(0, len(y_batch), 64)
                
                y_batch_arr = np.array(y_batch).reshape((-1, 1)) * action_batch_multiplier

                self.variables_dict = read_value_from_theta(l_theta, self.variables_dict, self.sess)
                
                feed_dict = {self.variables_dict["input_observation"]: observation_batch[sampling, :],
                             self.variables_dict["y_true"]: y_batch_arr[sampling, :], 
                             self.variables_dict["y_action"]: action_batch_multiplier[sampling, :]}
                self.sess.run(self.train_step, feed_dict=feed_dict)


                l_theta = assign_value_to_theta(l_theta, self.variables_dict, self.sess)

            if epsilon>0.01:
                epsilon -= 0.895/50000

        return


class master_worker(mp.Process):
    
    def __init__(self, T_max=100, nb_env=10, env_name="CartPole-v0", model_option={"n_hidden":1, "hidden_size":[10]}, 
                 verbose=False, **kwargs):
        import tensorflow as tf
        
        super(master_worker, self).__init__(**kwargs)
        self.T_max = T_max
        self.env = gym.make(env_name)
        self.nb_env = nb_env
        
        self.variables_dict = create_variable(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
        self.variables_dict["y"] = build_model(self.variables_dict, n_hidden=model_option["n_hidden"], 
                                               hidden_size=model_option["hidden_size"])

        self.loss, self.train_step = build_loss(self.variables_dict["y"], self.variables_dict)

    def run(self):
        global l_theta
        import tensorflow as tf

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        self.variables_dict = read_value_from_theta(l_theta, self.variables_dict, self.sess)
        saver.save(self.sess, './ptb_rnnlm.weights')

        epsilon = 0.01
        observation = self.env.reset()

        for i in range(self.nb_env):
            t = 0
            done = False
            while t<self.T_max:
                t += 1
                self.env.render()

                action = epsilon_greedy_policy(self.variables_dict, observation, epsilon, self.env, self.sess)

                observation, reward, done, info = self.env.step(action) 

                if done:
                    print("Environment completed in %s timesteps"%t)
                    observation = self.env.reset()
                    t += self.T_max
            if not done:
                print("Environment last %s timesteps"%t)
        return

def main(nb_process, T_max=5000,  model_option={"n_hidden":1, "hidden_size":[10]}, env_name="CartPole-v0"):
    global T, l_theta, l_theta_minus
    T = mp.Value('i', 0)
    
    l_theta = initialise(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
    l_theta_minus = initialise(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
    
    jobs = []
    for i in range(nb_process):
        print("Process %s starting"%i)
        job = slave_worker(T_max=T_max, model_option=model_option, env_name=env_name)
        job.start()
        jobs.append(job)
    
    t_init = time.time()
    while T.value<T_max:
        if time.time() - t_init > 1:
            print(T.value)
            t_init = time.time()
    print("Training completed")
    
    exemple = master_worker(T_max=T_max, model_option=model_option, env_name=env_name)
    exemple.run()
    
    """model.set_weights(theta.value)
    
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
    if len(args)>1:
        main(1, T_max=int(args[1]))
    else:
        main(3)
