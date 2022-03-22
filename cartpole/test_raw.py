import cartpole as cp
import gym

print("### TESTING THE RAW ALGORITHM WORKS/SAVING TRAINED Q TABLE ###")

algtype = "QL"    #Either QL or SARSA works
alpha = 0.1       #step parameter
gamma = 0.95       #discount factor
epsilon = 1    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
epsilon_decay_value = 0.99995
n = 60000         #number of episodes
cartpole_env = gym.envs.make("CartPole-v1")

test_env = cp.target_cartpole_agent(cartpole_env)
print("Using QL:")
result, Q = test_env.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n)

# #write Q Table to external file so I don't have to keep retraining for adversaries
# Q_file = open('trained_Q.p','wb')
# pickle.dump(Q , Q_file)
# Q_file.close()

# algtype = "SARSA"
# print("Using SARSA:")
# result, _ = test_env.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n)