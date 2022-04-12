################################
#Use the investigate_epsilon_resiliency.cartpole_epsilon_change() to perform multiple train-test cycles and average,
#how the epsilon delay (x-axis) affects the average reward (y-axis) for different environments (4 diff graphs)
#and on different attack frequencies (different lines on the graph).
#There are four graphs in the figure, one for each attacking environment.

import investigate_epsilon_resiliency as ier
import gym
import test_att_train_pattanaik as pattanaik
import test_att_train_naive_state as naive
import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

#####Note - we can change the algorithm below to use either commented out verison: depending on whether we want to
#attack during testing or during training. ####
train_or_test = "train"

algtype = "QL"       #Either QL or SARSA works
alpha = 0.1          #step parameter
gamma = 0.95         #discount factor
epsilon = 1          #exploration parameter (0 for pure exploitation, 1 for pure exploration)
epsilon_decay_value = 0.99995
n_train = 60000      #number of episodes
n_test = 1000      #number of times in each individual test
n_repeat_cycle = 4   #number of times to repeat the train-test cycles

Q_file = open('cartpole/trained_Q.p', 'rb')
Q = pickle.load(Q_file)   #obtained a Q-matrix estimate from previous training, for the pattanaik attack
Q_file.close()  

#Initialize cartpole environment from open AI gym
cartpole_env = gym.envs.make("CartPole-v1")

#Environments to test on att_freq variations, using the different exploitation delays
att_freq_arr =  np.array([3, 8, 20, 50, 100]) #Attack frequencies to try
exploit_delays = np.array([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]) #Exploit delays
delta = "MP"
pattanaik_minp_env = pattanaik.cartpole_adv_pattanaik_training(cartpole_env, att_freq_arr[0], Q, delta)        #min pertubation version
delta = "FP"
pattanaik_fullp_env = pattanaik.cartpole_adv_pattanaik_training(cartpole_env, att_freq_arr[0], Q, delta) #full perturbation version
delta = "MP"
naive_minp_env = naive.cartpole_adv_naive_training(cartpole_env, att_freq_arr[0], delta)             #full perturbation version
delta = "FP"
naive_fullp_env = naive.cartpole_adv_naive_training(cartpole_env, att_freq_arr[0], delta)             #full perturbation version
envs = [pattanaik_minp_env, pattanaik_fullp_env, naive_minp_env, naive_fullp_env]
env_labels = ["pattanaik_minp", "pattanaik_fullp", "naive_minp", "naive_fullp"]

#initialize altered target agent using this environment (if it's the training attack, we overwrite this anyway)
target_agent = ier.cartpole_epsilon_change(cartpole_env)

att_results = np.zeros((exploit_delays.size, att_freq_arr.size)) #Reward storing matrices for plotting later (co vectors of each environment, rows for att frequencies, 3rd dim for repeats)
fig, ax = plt.subplots(3,2) #set up graph so we plot 4 (one for each environment)

#Proceed to train cartpole, test environments, then repeat then take the average.
for i in range(len(envs)):
    for j in range(n_repeat_cycle):
        for k in range(exploit_delays.size):
            target_agent.set_env(cartpole_env) #before training, set the raw cartpole training environment.
            target_agent.train(algtype, alpha, gamma, epsilon, epsilon_decay_value, exploit_delays[k], n_train)
            for l in range(att_freq_arr.size):
                envs[i].set_att_freq(att_freq_arr[l]) #before testing, set the correct attack as the man-in-the-middle.
                target_agent.set_env(envs[i])
                att_results[k,l] = target_agent.test(n_test)

    att_results = att_results / n_repeat_cycle #average over cycles
    for j in range(att_freq_arr.size):
        i_axis = math.floor(i/2)
        j_axis = i%2
        ax[i_axis,j_axis].plot(exploit_delays, att_results[:,j], label=("Att Freq: " + str(att_freq_arr[j])))
        ax[i_axis,j_axis].set_title(env_labels[i] + " (train att)")
        ax[i_axis,j_axis].set_xlabel('Exploit Delay (iterations)')
        ax[i_axis,j_axis].set_ylabel('Average Reward Over {} Episodes'.format(n_test))
        ax[i_axis,j_axis].legend() 

plt.show()
