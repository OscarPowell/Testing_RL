#Testing changing the parameters of the epsilon decay and the delay for the decay
#on the raw environment (no adversary).

import investigate_epsilon_resiliency as ier
import gym
import pickle
import numpy as np
import matplotlib.pyplot as plt

#Raw target agent with no adversary (epsilon investigation version)
cartpole_env = gym.envs.make("CartPole-v1")
target_agent = ier.cartpole_epsilon_change(cartpole_env)

#Parameters of model
algtype = "QL"       #Either QL or SARSA works
alpha = 0.1          #step parameter
gamma = 0.95         #discount factor
epsilon = 1          #exploration parameter (0 for pure exploitation, 1 for pure exploration)
epsilon_decay_value = 0.99995
n_train = 60000      #number of episodes
n_test = 1000       #number of times in each individual test
n_repeat_cycle = 15   #number of times to repeat the train-test cycles
Q_file = open('cartpole/trained_Q.p', 'rb')
Q = pickle.load(Q_file)   #obtained a Q-matrix estimate from previous training, for the pattanaik attack
Q_file.close()  
exploit_delays = np.array([0, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000, 18000, 20000]) #Exploit delays

att_results = np.zeros((exploit_delays.size,)) #Reward storing matrices for plotting later (co vectors of each environment, rows for att frequencies, 3rd dim for repeats)
fig, ax = plt.subplots(1,1) #set up graph so we plot 4 (one for each environment)

#start at j to match the other file. repeat over a number of train-test cycles, trying out each exploit delay for each attack freq.
for j in range(n_repeat_cycle):
    print("Train-Test Cycle: " + str(n_repeat_cycle))
    for k in range(exploit_delays.size):
        print("Starting exploit delay: " + str(exploit_delays[k]))
        target_agent.train(algtype, alpha, gamma, epsilon, epsilon_decay_value, exploit_delays[k], n_train)
        att_results[k] = target_agent.test(n_test)

att_results = att_results/n_repeat_cycle
ax.plot(exploit_delays,att_results)
plt.show()