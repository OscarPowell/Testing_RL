#Perform the Pattanaik attack on an already trained model, using both QL and SARSA.
#Then compare the effect of this attack to a naive attack, running both at different frequencies.
#Also can compare the minimum perturbation vs max perturbation methods.
import gym
import cartpole as cp
import test_att_train_pattanaik as pattanaik
import test_att_train_naive_state as naive
import pickle
import numpy as np
import matplotlib.pyplot as plt

algtype = "QL"       #Either QL or SARSA works
alpha = 0.1          #step parameter
gamma = 0.95         #discount factor
epsilon = 1          #exploration parameter (0 for pure exploitation, 1 for pure exploration)
epsilon_decay_value = 0.99995
n_train = 60000      #number of episodes
n_test = 1000        #number of times in each individual test
n_repeat_cycle = 5   #number of times to repeat the train-test cycles

Q_file = open('cartpole/trained_Q.p', 'rb')
Q = pickle.load(Q_file)   #obtained a Q-matrix estimate from previous training, for the pattanaik attack
Q_file.close()  

#First train the agent's Q table and test it with test()
cartpole_env = gym.envs.make("CartPole-v1")
target_agent = cp.target_cartpole_agent(cartpole_env)      
target_agent.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n_train) #should store it's Q matrix from this.
raw_results = target_agent.test(n_test)
print("Raw results: " + str(raw_results))

#Environments to test
att_freq_arr =  np.array([1,2,3,5,8,10,20,50,100,200]) #Attack frequencies to try
delta = "MP"
pattanaik_minp_env = pattanaik.cartpole_adv_pattanaik_training(cartpole_env,att_freq_arr[0],Q, delta)        #min pertubation version
delta = "FP"                   
pattanaik_fullp_env = pattanaik.cartpole_adv_pattanaik_training(cartpole_env,att_freq_arr[0],Q, delta) #full perturbation version
delta = "MP"
naive_minp_env = naive.cartpole_adv_naive_training(cartpole_env,att_freq_arr[0],delta)             #full perturbation version
delta = "FP"
naive_fullp_env = naive.cartpole_adv_naive_training(cartpole_env,att_freq_arr[0],delta)             #full perturbation version
envs = [pattanaik_minp_env, pattanaik_fullp_env, naive_minp_env, naive_fullp_env]

att_results = np.zeros((att_freq_arr.size, len(envs))) #Reward storing matrices for plotting later (co vectors of each environment, rows for att frequencies, 3rd dim for repeats)
fig, ax = plt.subplots() #set up graph

#Proceed to train cartpole, test environments, then repeat then take the average.
for i in range(n_repeat_cycle):
    target_agent = cp.target_cartpole_agent(cartpole_env)      
    target_agent.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n_train)
#Plot the results of the naive attack and the pattanaik attack for each attack frequency
    for j in range(att_freq_arr.size):
        #Perform testing and plot the results
        map(lambda x: x.set_att_freq(att_freq_arr[j]), envs) #set the attack frequency for all environments
        for k in range(len(envs)):
            target_agent.set_env(envs[k])                 #run through the adversarial environments
            att_results[j,k] += target_agent.test(n_test) #test and return average reward
att_results = att_results / n_repeat_cycle #average over cycles

labels = ("Pattanaik, MP", "Pattanaik, FP", "Naive MP", "Naive, FP")
for i in range(len(envs)):
    ax.plot(att_freq_arr.reshape(att_freq_arr.size,1),att_results[:,i], label=labels[i])


ax.set_xlabel('Attack Spacing (iterations/attack)')
ax.set_ylabel('Average Reward Over {} Episodes'.format(n_test))
ax.set_title('Average Reward During Testing of QL on the cart-pole Environment, Naive vs Pattanaik Attacks')
ax.legend() 
plt.show()