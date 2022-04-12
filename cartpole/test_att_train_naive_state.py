########### Adv attack on TRAINING by RANDOM NOISE, changing the STATE recieved ##########
import cartpole as cp
import math
import numpy as np
import matplotlib.pyplot as plt
import gym
import random

#assign a random state to feed back to the training algorithm, knowing the observation space being between -2.4 to 2.4 for cart pole position, -.148 to .148 pole angle, and infinite speed/pole speed

#When initializing, env is the environment being trained on, att_freq is the gap between attacks in iterations, delta is the perturbation string ("MP" for nearest neighbours, "FP" for using the whole observation space) 
class cartpole_adv_naive_training():
    def __init__(self, env, att_freq, delta):
        self.env = env
        self.action_space = env.action_space
        self.step_counter = 0
        self.att_freq = att_freq
        self.observation_space = env.observation_space
        self.delta = delta
        self.np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1]) 
        print("Initialized random-state adv agent with att_freq: ", att_freq)
    
    def set_att_freq(self, att_freq):
        self.att_freq = att_freq

    def set_env(self, env):
        self.env = env

    def get_discrete_state(self, state):
        discrete_state = state/(self.np_array_win_size)+ np.array([15,10,1,10])
        return tuple(discrete_state.astype(int))

        #Reverses the get_discrete state equation essentially.
    def get_continuous_state(self, state):
        return (state-np.array([15,10,1,10]))*self.np_array_win_size

    #Every step_counter iterations, it will step the environment but provide a random state back to the training agent.
    #The random state only affects the cart position and the pole position for now. Speeds are set to 0
    def step(self, A):
        self.step_counter += 1
        if self.step_counter % self.att_freq == 0:
            if self.delta == "FP":
                _ , R, done, _ = self.env.step(A) #get the normal values of R and done
                observation_space_actual = [2.4, 2.5, 12*2*math.pi/360, 1] #bounds for state
                Snew = (np.random.rand(4)*2 - 1) * observation_space_actual #Generate random states within correct range.
                return Snew, R, done, _  
            else:
                S_actual, R, done, _ = self.env.step(A) #get the normal values of R and done
                #Will have to convert to a discrete state for Q matrix indices, then convert back to original continuous scale to return to the target agent. 
                S_actual = self.get_discrete_state(S_actual)

                #find neighbouring states by looping through forward and backward perturbations on each dimension, choosing a random one.
                Snew = self.generate_neighbour(S_actual) 
                return self.get_continuous_state(Snew), R, done, _  
        else:
            return self.env.step(A)

    def reset(self):
        return self.env.reset()
    
    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def generate_neighbour(self, S):
        #Perturbs each dim by -1,0 and +1 and gets all permutations to get all immediate neighbours.
        S_i = S[0]
        S_j = S[1]
        S_k = S[2]
        S_l = S[3]
        return (random.randint(S_i-1,S_i+1),random.randint(S_j-1,S_j+1),random.randint(S_k-1,S_k+1),random.randint(S_l-1,S_l+1))

# algtype = "QL"    #Either QL or SARSA works
# alpha = 0.1       #step parameter
# gamma = 0.95       #discount factor
# epsilon = 1    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
# epsilon_decay_value = 0.99995
# n = 60000         #number of episodes
# cartpole_env = gym.envs.make("CartPole-v1")

# print("Using QL, random state attack on training, for different attack frequencies:")
# att_freq_arr = [1,2,3,5,8,10,20,50]
# plot_episodes = np.linspace(0, n, int(n/1000 + 1)) #every 1000 episodes we take a data point
# fig, ax = plt.subplots()
# for att_freq in att_freq_arr:
#     adv_env = cartpole_adv_randomstate_training(cartpole_env, att_freq) #set up the adversarial environment for the man-in-the-middle attack, with the attack frequency
#     target_agent = cp.target_cartpole_agent(adv_env)                       #set up target agent with the adversary in
#     plot_rewards, _ = target_agent.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n) #do training and return results to plot later
#     thislabel = 'Att Freq: ' + str(att_freq)
#     ax.plot(plot_episodes,plot_rewards, label=thislabel)
# #Do one without any attack to compare
# target_agent = cp.target_cartpole_agent(cartpole_env)
# plot_rewards = target_agent.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n)
# thislabel = 'No attack'
# ax.plot(plot_episodes,plot_rewards, label=thislabel)

# ax.set_xlabel('Episode Number')
# ax.set_ylabel('Average Reward Per 1000 Episodes')
# ax.set_title('Average Reward During Training with QL on cart-pole Environment, Random State Attack, Varying Frequency')
# ax.legend()
# plt.show()
#Try to plot over the 60000 episodes, while changing the attack frequency to see the effect of the attack.

