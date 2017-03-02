
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
    variables_dict["W1" + name + "_assign_ph"] = tf.placeholder(tf.float32, shape=[input_size, hidden_size[0]], 
        name="W1"+name+"_assign_ph")
    variables_dict["W1" + name + "_assign"] = tf.assign(variables_dict["W1" + name], 
        variables_dict["W1" + name + "_assign_ph"])
    variables_dict["W1_grad_ph_policy" + name] = tf.placeholder(tf.float32, shape=[input_size, hidden_size[0]],
        name="W1_grad_ph_policy" + name)
    variables_dict["W1_grad_ph_vf" + name] = tf.placeholder(tf.float32, shape=[input_size, hidden_size[0]],
        name="W1_grad_ph_vf" + name)
    
    variables_dict["b1" + name] = bias_variable((1, hidden_size[0]), name="b1" + name)
    variables_dict["b1" + name + "_assign_ph"] = tf.placeholder(tf.float32, shape=[1, hidden_size[0]], 
        name="b1"+name+"_assign_ph")
    variables_dict["b1" + name + "_assign"] = tf.assign(variables_dict["b1" + name], 
        variables_dict["b1" + name + "_assign_ph"])
    variables_dict["b1_grad_ph_policy" + name] = tf.placeholder(tf.float32, shape=[1, hidden_size[0]],
        name="b1_grad_ph_policy" + name)
    variables_dict["b1_grad_ph_vf" + name] = tf.placeholder(tf.float32, shape=[1, hidden_size[0]],
        name="b1_grad_ph_vf" + name)

    for i in range(n_hidden-1):
        variables_dict["W"+str(i+2) + name] = weight_variable([hidden_size[i], hidden_size[i+1]], 
            name="W"+str(i+2) + name)
        variables_dict["W"+str(i+2) + name + "_assign_ph"] = tf.placeholder(tf.float32, 
            shape=[hidden_size[i], hidden_size[i+1]], name="W"+str(i+2)+name+"_assign_ph")
        variables_dict["W"+str(i+2) + name + "_assign"] = tf.assign(variables_dict["W"+str(i+2) + name], 
            variables_dict["W"+str(i+2) + name + "_assign_ph"])
        variables_dict["W"+str(i+2) + "_grad_ph_policy" + name] = tf.placeholder(tf.float32, shape=[hidden_size[i], hidden_size[i+1]], 
            name="W"+str(i+2) + "_grad_ph_policy" + name)
        variables_dict["W"+str(i+2) + "_grad_ph_vf" + name] = tf.placeholder(tf.float32, shape=[hidden_size[i], hidden_size[i+1]], 
            name="W"+str(i+2) + "_grad_ph_vf" + name)
        

        variables_dict["b"+str(i+2) + name] = bias_variable((1, hidden_size[i+1]), name="b"+str(i+2) + name)
        variables_dict["b"+str(i+2) + name + "_assign_ph"] = tf.placeholder(tf.float32, 
            shape=[1, hidden_size[i+1]], name="b"+str(i+2)+name+"_assign_ph")
        variables_dict["b"+str(i+2) + name + "_assign"] = tf.assign(variables_dict["b"+str(i+2) + name], 
            variables_dict["b"+str(i+2) + name + "_assign_ph"])
        variables_dict["b"+str(i+2) + "_grad_ph_policy" + name] = tf.placeholder(tf.float32, shape=[1, hidden_size[i+1]],
            name="b"+str(i+2) + "_grad_ph_policy" + name)
        variables_dict["b"+str(i+2) + "_grad_ph_vf" + name] = tf.placeholder(tf.float32, shape=[1, hidden_size[i+1]],
            name="b"+str(i+2) + "_grad_ph_vf" + name)

    variables_dict["Wo_policy" + name] = weight_variable([hidden_size[-1], output_size], name="Wo_policy" + name)
    variables_dict["Wo_policy" + name + "_assign_ph"] = tf.placeholder(tf.float32, shape=[hidden_size[-1], output_size], 
        name="Wo_policy"+name+"_assign_ph")
    variables_dict["Wo_policy" + name + "_assign"] = tf.assign(variables_dict["Wo_policy" + name], 
        variables_dict["Wo_policy" + name + "_assign_ph"])
    variables_dict["Wo_policy_grad_ph_policy" + name] = tf.placeholder(tf.float32, shape=[hidden_size[-1], output_size],
        name="Wo_policy_grad_ph_policy" + name)

    variables_dict["bo_policy" + name] = bias_variable((1, output_size), name="bo_policy" + name)
    variables_dict["bo_policy" + name + "_assign_ph"] = tf.placeholder(tf.float32, shape=[1, output_size], 
        name="bo_policy"+name+"_assign_ph")
    variables_dict["bo_policy" + name + "_assign"] = tf.assign(variables_dict["bo_policy" + name], 
        variables_dict["bo_policy" + name + "_assign_ph"])
    variables_dict["bo_policy_grad_ph_policy" + name] = tf.placeholder(tf.float32, shape=[1, output_size],
        name="bo_policy_grad_ph_policy" + name)

    variables_dict["Wo_vf" + name] = weight_variable([hidden_size[-1], 1], name="Wo_vf" + name)
    variables_dict["Wo_vf" + name + "_assign_ph"] = tf.placeholder(tf.float32, shape=[hidden_size[-1], 1], 
        name="Wo_vf"+name+"_assign_ph")
    variables_dict["Wo_vf" + name + "_assign"] = tf.assign(variables_dict["Wo_vf" + name], 
        variables_dict["Wo_vf" + name + "_assign_ph"])
    variables_dict["Wo_vf_grad_ph_vf" + name] = tf.placeholder(tf.float32, shape=[hidden_size[-1], 1],
        name="Wo_vf_grad_ph_vf" + name)

    variables_dict["bo_vf" + name] = bias_variable((1, 1), name="bo_vf" + name)
    variables_dict["bo_vf" + name + "_assign_ph"] = tf.placeholder(tf.float32, shape=[1, 1], 
        name="bo_vf"+name+"_assign_ph")
    variables_dict["bo_vf" + name + "_assign"] = tf.assign(variables_dict["bo_vf" + name], 
        variables_dict["bo_vf" + name + "_assign_ph"])
    variables_dict["bo_vf_grad_ph_vf" + name] = tf.placeholder(tf.float32, shape=[1, 1],
        name="bo_vf_grad_ph_vf" + name)

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
    
    vf = tf.matmul(y, variables_dict["Wo_vf" + name]) + variables_dict["bo_vf" + name]
    actions = tf.nn.softmax(tf.matmul(y, variables_dict["Wo_policy" + name]) + variables_dict["bo_policy" + name])

    return actions, vf

def build_loss(vf, actions, variables_dict, alpha_reg=0, beta_reg=0.01):
    import tensorflow as tf
    loss_list_policy = tf.log(tf.matmul(actions, variables_dict["y_action"]))
    loss_policy = tf.reduce_mean(loss_list_policy)

    loss_list_vf = tf.nn.l2_loss(vf - variables_dict["y_true"])
    loss_vf = tf.reduce_mean(loss_list_vf)

    # l1_reg = 0
    # l2_reg = 0

    # keys = variables_dict.keys()
    # keys.sort()
    # keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "actions", "values"]) & (key[-3:] != "_ph") & \
    #         (key[-7:] != "_assign")]
    # for key in keys:
    #     l1_reg += tf.reduce_sum(tf.abs(variables_dict[key]))
    #     l2_reg += tf.nn.l2_loss(variables_dict[key])

    # loss += alpha_reg * l1_reg + beta_reg * l2_reg

    return loss_policy, loss_vf

def compute_gradients(loss_policy, loss_vf, vf, variables_dict):
    import tensorflow as tf
    grads = {}
    
    keys = variables_dict.keys()
    keys = [key for key in keys if (key not in ["input_observation", "y_true", "y_action", "actions", "values"]) & ("ph" not in key) & \
            ("assign" not in key)]
    common_keys = [key for key in keys if ("policy" not in key) & ("vf" not in key)]
    policy_keys = [key for key in keys if "policy" in key]
    vf_keys = [key for key in keys if "vf" in key]
    for key in common_keys:
        grads[key + "_grad_ph_policy"] = tf.gradients(loss_policy, [variables_dict[key]])[0] * (variables_dict["y_true"] - vf)
        grads[key + "_grad_ph_vf"] = tf.gradients(loss_vf, [variables_dict[key]])[0]
    for key in policy_keys:
        grads[key + "_grad_ph_policy"] = tf.gradients(loss_policy, [variables_dict[key]])[0] * (variables_dict["y_true"] - vf)
    for key in vf_keys:
        grads[key + "_grad_ph_vf"] = tf.gradients(loss_vf, [variables_dict[key]])[0]

    return grads

def build_train_step(variables_dict, learning_rate):
    import tensorflow as tf

    keys = variables_dict.keys()
    keys = [key for key in keys if (key not in ["input_observation", "y_true", "y_action", "actions", "values"]) & ("ph" not in key) & \
            ("assign" not in key)]
    common_keys = [key for key in keys if ("policy" not in key) & ("vf" not in key)]
    policy_keys = [key for key in keys if "policy" in key]
    vf_keys = [key for key in keys if "vf" in key]
    updates = []
    for key in common_keys:
        updates.append((variables_dict[key + "_grad_ph_policy"], variables_dict[key]))
        updates.append((variables_dict[key + "_grad_ph_vf"], variables_dict[key]))
    for key in policy_keys:
        updates.append((variables_dict[key + "_grad_ph_policy"], variables_dict[key]))
    for key in vf_keys:
        updates.append((variables_dict[key + "_grad_ph_vf"], variables_dict[key]))

    opt = tf.train.RMSPropOptimizer(learning_rate, decay=0.99, momentum=0.5, centered=True)
    train_step = opt.apply_gradients(updates)
    return train_step

def best_action(variables_dict, observation, sess):
    import tensorflow as tf
    feed_dic = {variables_dict["input_observation"]: observation.reshape((1, -1))}
    #print("Je passe")
    actions = sess.run(variables_dict["actions"], feed_dict=feed_dic)
    
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

    return np.argmax(actions)

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
    keys = [ key for key in keys if (key not in ["input_observation", "y_true", "y_action", "actions", "values"]) & ("ph" not in key) & \
            ("assign" not in key)]
    for i, key in enumerate(keys):
        l_theta[i] = sess.run(variables_dict[key])
    return l_theta
    
def read_value_from_theta(variables_dict, sess):
    global l_theta
    import tensorflow as tf
    keys = variables_dict.keys()
    keys.sort()
    keys = [key for key in keys if (key not in ["input_observation", "y_true", "y_action", "actions", "values"]) & ("ph" not in key) & \
            ("assign" not in key)]
    for i, key in enumerate(keys):
        feed_dict = {variables_dict[key + "_assign_ph"]: l_theta[i]}
        sess.run(variables_dict[key + "_assign"], feed_dict=feed_dict)
    return variables_dict

def initialise(input_size=4, output_size=2, n_hidden=2, hidden_size=[128, 64]):
    l_theta = mp.Manager().list()
    
    shapes = [(input_size, hidden_size[0])]
    for i in range(n_hidden - 1):
        shapes.append((hidden_size[i], hidden_size[i+1]))
    shapes.append((hidden_size[-1], output_size))
    shapes.append((hidden_size[-1], 1))

    shapes.append((1, hidden_size[0]))
    for i in range(n_hidden - 1):
        shapes.append((1, hidden_size[i+1]))
    shapes.append((1, output_size))
    shapes.append((1, 1))
    
    for i, shape in enumerate(shapes):
        l_theta.append(np.random.uniform(low=-0.01, high=0.01, size=shape))
        # l_theta[i].value = np.random.uniform(low=-0.01, high=0.01, size=shape)
        
    return l_theta


# In[6]:

class slave_worker(mp.Process):
    
    def __init__(self, T_max=100, t_max=5, gamma=0.9, learning_rate=0.001, 
                   env_name="CartPole-v0", model_option={"n_hidden":1, "hidden_size":[10]}, 
                   verbose=False, policy=None, epsilon_ini=0.9, **kwargs):
        super(slave_worker, self).__init__(**kwargs)
        self.T_max = T_max
        self.t_max = t_max
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
        self.variables_dict["actions"], self.variables_dict["values"] = build_model(self.variables_dict,
                                                                                    n_hidden=model_option["n_hidden"], 
                                                                                    hidden_size=model_option["hidden_size"])
        self.loss_policy, self.loss_vf = build_loss(self.variables_dict["values"], self.variables_dict["actions"],
                                                    self.variables_dict)
        self.gradients = compute_gradients(self.loss_policy, self.loss_vf, self.variables_dict["values"], self.variables_dict)
        self.train_step = build_train_step(self.variables_dict, learning_rate=learning_rate)
            
        
    def run(self):
        import tensorflow as tf
        global T, l_theta

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        epsilon = self.epsilon_ini
        t = 0
        x_batch = 0
        y_batch = []
        nb_env = 0

        observation = self.env.reset()

        while T.value<self.T_max:

            # if T.value %500 == 0:
            #     print(T.value)
            t = 0
            t_init = t
            done = False

            observation_batch = observation.reshape((1, -1))

            reward_batch = []
            action_batch = []

            d_theta = {}
            keys = self.variables_dict.keys()
            keys = [key for key in keys if (key not in ["input_observation", "y_true", "y_action", "actions", "values"]) & ("ph" not in key) & \
                    ("assign" not in key)]
            common_keys = [key for key in keys if ("policy" not in key) & ("vf" not in key)]
            policy_keys = [key for key in keys if "policy" in key]
            vf_keys = [key for key in keys if "vf" in key]
            for key in common_keys:
                d_theta[key + "_grad_ph_policy"] = 0
                d_theta[key + "_grad_ph_vf"] = 0
            for key in policy_keys:
                d_theta[key + "_grad_ph_policy"] = 0
            for key in vf_keys:
                d_theta[key + "_grad_ph_vf"] = 0

            self.variables_dict = read_value_from_theta(self.variables_dict, self.sess)

            while (not done) & (t-t_init<=self.t_max):
            
                if self.verbose:
                    self.env.render()
                    if T.value%5000 == 0:
                        print("T = %s"%T.value)

                action = epsilon_greedy_policy(self.variables_dict, observation, epsilon, self.env, self.sess, self.policy)

                observation, reward, done, info = self.env.step(action) 

                reward_batch.append(reward)
                action_batch.append(action)
                observation_batch = np.vstack((observation.reshape((1, -1)), observation_batch))
                

                #if t - t_init > 200:
                #    done = True

                if done:
                    nb_env += 1
                    observation = self.env.reset()
                
                with T.get_lock():
                    T.value += 1
                
                t += 1

                if epsilon>0.01:
                    epsilon -= (self.epsilon_ini - 0.01)/50000
            
            if done:
                R = 0
            else:
                R = self.sess.run(self.variables_dict["values"],
                    feed_dict={self.variables_dict["input_observation"]: observation.reshape((1, -1))})[0,0]
            
            action_batch.reverse()
            for i in range(t - 1, t_init - 1, -1):
                R = reward_batch[i] + self.gamma * R
                feed_dict = {self.variables_dict["input_observation"]: observation_batch[i,:].reshape((1, -1)),
                             self.variables_dict["y_true"]: np.array([[R]]), 
                             self.variables_dict["y_action"]: np.eye(2)[[action_batch[i]]].T}
                for key in d_theta.keys():
                    d_theta[key] += self.sess.run(self.gradients[key], feed_dict=feed_dict)
            
            feed_dict = {self.variables_dict[key]: d_theta[key] for key in d_theta.keys()}
            self.variables_dict = read_value_from_theta(self.variables_dict, self.sess)
            self.sess.run(self.train_step, feed_dict=feed_dict)

            l_theta = assign_value_to_theta(self.variables_dict, self.sess)

        return


class master_worker(mp.Process):
    
    def __init__(self, T_max=1000, t_max=200, env_name="CartPole-v0", model_option={"n_hidden":1, "hidden_size":[10]}, 
                 verbose=False, n_sec_print=10, goal=195, len_history=100, Itarget=15, **kwargs):
        import tensorflow as tf
        
        super(master_worker, self).__init__(**kwargs)
        self.T_max = T_max
        self.t_max = t_max
        self.env = gym.make(env_name)
        self.nb_env = 0
        self.Itarget = Itarget
        
        self.variables_dict = create_variable(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
        self.variables_dict["actions"], self.variables_dict["values"] = build_model(self.variables_dict,
                                                                                    n_hidden=model_option["n_hidden"], 
                                                                                    hidden_size=model_option["hidden_size"])

        self.history = [0 for i in range(len_history)]
        self.goal = goal
        self.max_mean = 0
        self.current_mean = 0
        self.last_T = 0
        self.n_sec_print = n_sec_print

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
        global l_theta
        import tensorflow as tf
        t_taken = time.time()

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
            if time.time() - t_init > self.n_sec_print:
                print("T = %s"%T.value)
                print("Max mean = %s"%self.max_mean)
                print("Current mean = %s"%self.current_mean)
                t_init = time.time()

            t = 0
            self.nb_env += 1
            while t<self.t_max:

                t += 1
                action = epsilon_greedy_policy(self.variables_dict, observation, epsilon, self.env, self.sess)

                observation, reward, done, info = self.env.step(action) 

                if done:
                    #print("Environment completed in %s timesteps"%t)
                    observation = self.env.reset()
                    self.add_history(t)
                    t += self.T_max
            if not done:
                observation = self.env.reset()
                #print("Environment last %s timesteps"%t)
                self.add_history(t)
                self.last_T = T.value
            else:
                self.variables_dict = read_value_from_theta(self.variables_dict, self.sess)

        print("Training completed")
        saver.save(self.sess, './end_training.weights')
        print("T final = %s"%self.last_T)
        print("Done in %s environments"%(self.nb_env-100))
        print("Done in %s seconds"%(time.time() - t_taken))
        T.value += self.T_max

        observation = self.env.reset()
        #self.variables_dict = read_value_from_theta(self.variables_dict, self.sess)

        for i in range(10):
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

def main(nb_process, T_max=5000,  model_option={"n_hidden":1, "hidden_size":[10]}, env_name="CartPole-v0",
         t_max=5, Itarget=15, gamma=0.9, learning_rate=0.001, verbose=False, policy=False, 
         epsilon_ini=0.9, several_eps=True, n_sec_print=10, master=False, goal=195, len_history=100):
    global T, l_theta

    T = mp.Value('i', 0)
    
    l_theta = initialise(n_hidden=model_option["n_hidden"], hidden_size=model_option["hidden_size"])
    
    jobs = []
    if policy:
        policies = create_2D_policies(nb_process)
    else:
        policies = [None for i in range(nb_process)]
    if several_eps:
        epsilons = create_list_epsilon(nb_process)
    else:
        epsilons = [epsilon_ini for i in range(nb_process)]
    verboses = [False for i in range(nb_process)]
    if master:
        verboses[0] = True
    for i in range(nb_process):
        print("Process %s starting"%i)
        job = slave_worker(T_max=T_max, model_option=model_option, env_name=env_name, 
            policy=policies[i], epsilon_ini=epsilons[i], t_max=t_max, gamma=gamma, 
            learning_rate=learning_rate, verbose=verboses[i])
        job.start()
        jobs.append(job)
    
    
    exemple = master_worker(T_max=T_max, t_max=200, model_option=model_option, env_name=env_name, 
                            n_sec_print=n_sec_print, goal=goal, len_history=len_history, Itarget=Itarget)
    exemple.start()
    exemple.join()

if __name__=="__main__":
    import sys
    args = sys.argv
    if len(args)>2:
        main(int(args[1]), T_max=int(args[2]), model_option={"n_hidden":2, "hidden_size":[128, 128]})
    else:
        main(3, 50000)
