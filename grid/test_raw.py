import grid_world as grid
import numpy as np
import matplotlib.pyplot as plt

algtypes = ("QL", "SARSA")    #Either QL or SARSA works
alpha = 0.5       #step parameter
gamma = 0.9       #discount factor
epsilon = 0.05    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
n = 1600         #number of episodes 
fr_gap = 0        #gap between iterations for false rewards provided (0 for no false rewards)
n_test = 1600
n_cycle_repeats = 30

grid_env = grid.grid_env()
test_env = grid.target_grid_agent(grid_env)               #pass in grid environment to the training agent

plot_episodes = np.linspace(0, n, int(n/50 + 1)) #every 1000 episodes we take a data point
plot_training_rewards = np.zeros((int(n/50 + 1), len(algtypes))) #plot rewards per 1000 episodes, for each algtype
testing_rewards = np.zeros((grid_env.observation_space,len(algtypes))) #stores the matrices which show average reward for starting in each position
for i in range(n_cycle_repeats):
    for j in range(len(algtypes)):
        rewards, _ = test_env.train(algtypes[j], alpha, gamma, epsilon, n, fr_gap) #train agent and store Q table
        plot_training_rewards[:,j] += rewards
        testing_rewards[:,j] = testing_rewards[:,j] + test_env.test(n_test)
#take the average of the repeats
plot_training_rewards = plot_training_rewards / n_cycle_repeats 
testing_rewards = testing_rewards / n_cycle_repeats
print("Testing Rewards, QL: {}, SARSA: {}".format(testing_rewards[0],testing_rewards[1]))

fig, ax = plt.subplots() #set up graph
for i in range(len(algtypes)):
    ax.plot(plot_episodes, plot_training_rewards[:,i], label=algtypes[i])

ax.set_xlabel('Iterations')
ax.set_ylabel('Average Reward Over {} Training Repeats'.format(n_cycle_repeats))
ax.set_title('Average Reward During Training of QL vs SARSA on the grid Environment')
ax.legend() 
plt.show()
