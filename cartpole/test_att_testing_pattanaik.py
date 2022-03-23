#Perform the Pattanaik attack on an already trained model, using both QL and SARSA.
import gym
import cartpole as cp
import test_att_train_pattanaik as pattanaik
import pickle

algtype = "QL"    #Either QL or SARSA works
alpha = 0.1       #step parameter
gamma = 0.95       #discount factor
epsilon = 1    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
epsilon_decay_value = 0.99995
n_train = 60000         #number of episodes
att_freq = 3
n_test = 1000

Q_file = open('cartpole/trained_Q.p', 'rb')
Q = pickle.load(Q_file)      #obtained a Q-matrix estimate from previous training 
Q_file.close()  

#First train the agent's Q table and test it with test()
cartpole_env = gym.envs.make("CartPole-v1")
target_agent = cp.target_cartpole_agent(cartpole_env)
target_agent.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n_train) #should store it's Q matrix from this.
raw_results = target_agent.test(n_test)
print("\nRaw agent results:",raw_results)

adv_env = pattanaik.cartpole_adv_pattanaikrandom_training(cartpole_env,att_freq,Q)
target_agent.set_env(adv_env)
result = target_agent.test(n_test)

print("\nAttacked agent results with full space:",result)

adv_env = pattanaik.cartpole_adv_pattanaik_training(cartpole_env,att_freq,Q)
target_agent.set_env(adv_env)
result = target_agent.test(n_test)

print("\nAttacked agent results with min perturbation:",result)


