import threading
import numpy as np

from numpy.random import binomial

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Reshape, Input
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import RMSprop, SGD
from keras.callbacks import ModelCheckpoint

import gym
import numpy as np
import time

from multiprocessing import *



def build_network(env, hidden_size=64):
    # create
    inputModel = Input(shape=(1 + env.observation_space.shape[0],))
    probs = Dense(hidden_size, activation='relu')(inputModel)
    probs = Dense(hidden_size, activation='relu')(probs)
    probs = Dense(1, activation='linear')(probs)
    model = Model(input=inputModel, output=probs)
    # TODO share
    rmsprop = RMSprop(lr=0.001)
    model.compile(optimizer=rmsprop, loss='mse')

    return (model)

#Weighted choice among the len(weights) classes.
#Based on the implementation : http://stackoverflow.com/a/3679747

##input : weights
#weights is a list of floats

##output : class
#class is an integer from 0 to len(weights)-1

def weighted_choice(weights):
    total = sum(w for w in weights)
    proba = np.array(weights).astype(float) / total
    r = np.random.uniform(0, 1)
    upto = 0.
    for c, p in enumerate(proba):
        if upto + p >= r:
            return c
        upto += p

def explorationGreedyPolicy(q_values, epsilon):
    isExploration = binomial(1,epsilon)
    if isExploration:
        action = weighted_choice([1. for value in q_values])
    else:
        action = np.argmax(q_values)
    return action

def getQValues(q_network, state, actionSpace):
    q_values = [q_network.predict(np.append([action], state).reshape((1,-1))) for action in actionSpace]
    return(q_values)

env = gym.make('CartPole-v0')

listOfAsyncTimes = [3, 5, 7, 11]
#

#
hidden_size = 64
#
target_q_network = build_network(env, hidden_size)
#q_network = build_network(env, hidden_size)
theta=target_q_network.get_weights()
theta_minus = target_q_network.get_weights()

T=0
Tmax=5000

def onestepdqlearn(thread_id,iAsyncUpdate,saveCountId=0,env_name='CartPole-v0',epsilon=1.,iTarget=17):
    print('Starting thread number '+str(thread_id))
    global theta_minus, theta, T, Tmax, hidden_size

    env = gym.make(env_name)
    actionSpace = range(env.action_space.n)

    q_network = build_network(env, hidden_size)
    target_q_network = build_network(env, hidden_size)
    target_q_network.set_weights(theta_minus)

    checkpoint = 0

    saveTime = 100

    time.sleep(3*thread_id)
    if checkpoint > 0:
        q_network.load_weights('temp/model-%d.h5' % (checkpoint,))
        target_q_network.set_weights(q_network.get_weights())

    #####################################

    gamma = 0.99999

    # Initialize thread step counter
    t = 0

    # Get initial state s
    meanTimeTarget = 0
    state = env.reset()
    timeEpisode = 0

    y_batch = []
    x_batch = []

    countEpisodeTarget = 0.
    oldMeanTimeTarget = 0
    if thread_id == saveCountId:
        saveCountEpisodes = open("countEpisodesOneStep.txt",'w')


    while T <= Tmax:

        timeEpisode += 1

        q_network.set_weights(theta)
        q_values = getQValues(q_network, state, actionSpace)
        action = explorationGreedyPolicy(q_values, epsilon)
        newState, reward, done, info = env.step(actionSpace[action])

        if done:
            y = reward
            countEpisodeTarget += 1
            meanTimeTarget = (meanTimeTarget * (countEpisodeTarget - 1) + timeEpisode) / countEpisodeTarget
            newState=env.reset()

            #Save count episodes for a given thread_id (early stop safe)
            if thread_id == saveCountId:
                saveCountEpisodes.write("%s\n" % timeEpisode)
                saveCountEpisodes.flush()
            timeEpisode = 0
            done=False
        else:
            target_q_network.set_weights(theta_minus)
            target_q_values = getQValues(target_q_network, newState, actionSpace)
            # gamma = preference pour le present
            y = reward + gamma * np.max(target_q_values)

        # Accumulate gradients wrt theta
        y_batch.append(y)
        x_t = np.append([action], [state]).tolist()
        x_batch.append(x_t)

        state = newState

        T += 1
        t += 1

        if T % iTarget == 0:
            print("Thread "+str(thread_id)+" - meanTime before target change : " + str(meanTimeTarget))
            countEpisodeTarget = 0.
            theta_minus=q_network.get_weights()
            target_q_network.set_weights(theta_minus)
            print(epsilon)
            if ((thread_id==saveCountId)&(T%100==0)):
            #if oldMeanTimeTarget <= meanTimeTarget:
                checkpoint += 1
                target_q_network.save_weights('temp/model-%d.h5' % (checkpoint,), overwrite=True)
                print('checkpoint')
            epsilon *= np.exp(-0.00001)

        if (t % iAsyncUpdate == 0) or done:
            q_network.set_weights(theta)
            q_network.train_on_batch(np.array(x_batch).reshape((-1, len(x_t))), np.array(y_batch).reshape((-1, 1)))
            theta=q_network.get_weights()
            # Clear gradients
            y_batch = []
            x_batch = []
    if thread_id == saveCountId:
        saveCountEpisodes.close()


def main():
	threads = [threading.Thread(target=onestepdqlearn,args=(thread_id,listOfAsyncTimes[thread_id])) for thread_id in range(cpu_count())]
	for thr in threads:
		thr.start()
	for thr in threads:
		thr.join()


if __name__ == '__main__':
	main()
