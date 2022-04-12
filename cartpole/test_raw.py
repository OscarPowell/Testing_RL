### TESTING THE RAW ALGORITHM WORKS AND COMPARING QL VS SARSA WHEN TRAINING AND TESTING###"
import cartpole as cp
import gym
import numpy as np
import matplotlib.pyplot as plt

algtypes = ("QL","SARSA")    #Either QL or SARSA works
alpha = 0.1       #step parameter
gamma = 0.95       #discount factor
epsilon = 1    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
epsilon_decay_value = 0.99995
n = 60000         #number of episodes
n_cycle_repeats = 1
n_test = 5000

cartpole_env = gym.envs.make("CartPole-v1")
test_env = cp.target_cartpole_agent(cartpole_env)

plot_episodes = np.linspace(0, n, int(n/1000 + 1)) #every 1000 episodes we take a data point
plot_training_rewards = np.zeros((int(n/1000 + 1), len(algtypes))) #plot rewards per 1000 episodes, for each algtype
testing_rewards = np.zeros((len(algtypes)))
for i in range(n_cycle_repeats):
    for j in range(len(algtypes)):
        rewards, _ = test_env.train(algtypes[j],alpha,gamma,epsilon,epsilon_decay_value,n)
        plot_training_rewards[:,j] += rewards
        testing_rewards[j] += test_env.test(n_test)
#take the average of the repeats
plot_training_rewards = plot_training_rewards / n_cycle_repeats 
testing_rewards = testing_rewards / n_cycle_repeats
print("Testing Rewards, QL: {}, SARSA: {}".format(testing_rewards[0],testing_rewards[1]))

fig, ax = plt.subplots() #set up graph
for i in range(len(algtypes)):
    ax.plot(plot_episodes, plot_training_rewards[:,i], label=algtypes[i])

ax.set_xlabel('Iterations')
ax.set_ylabel('Average Reward Over {} Training Repeats'.format(n_cycle_repeats))
ax.set_title('Average Reward During Training of QL vs SARSA on the cartpole Environment')
ax.legend() 
plt.show()

# #write Q Table to external file so I don't have to keep retraining for adversaries
# Q_file = open('trained_Q.p','wb')
# pickle.dump(Q , Q_file)
# Q_file.close()