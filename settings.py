# coding: utf-8
from utils import initialise, initialise_a3c
import multiprocessing as mp

def init(algo="nstep", n_hidden=1, hidden_size=[16],input_size=4, output_size=2):
	if algo == "a3c":
		global T, l_theta
	else:
		global T, l_theta, l_theta_minus

	T = mp.Value('i', 0)

	if algo == "a3c":
		l_theta = initialise_a3c(n_hidden=n_hidden, hidden_size=hidden_size, input_size=input_size, 
							 	 output_size=output_size)
	else:
		l_theta = initialise(n_hidden=n_hidden, hidden_size=hidden_size, input_size=input_size, 
							 output_size=output_size)
		l_theta_minus = initialise(n_hidden=n_hidden, hidden_size=hidden_size, input_size=input_size, 
								   output_size=output_size)
