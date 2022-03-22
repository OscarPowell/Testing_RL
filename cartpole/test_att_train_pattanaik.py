########### Adv Attack on TRAINING - PATTANAIK ET AL VERSION ########
#Using knowledge of the Q function
from sys import set_coroutine_origin_tracking_depth
import gym
import cartpole as cp
import numpy as np
import pickle
import matplotlib.pyplot as plt
import test_att_train_naive_state as naive_attack   

class cartpole_adv_pattanaik_training():
    def __init__(self, env, att_freq, Q):
        self.env = env
        self.action_space = env.action_space
        self.step_counter = 0
        self.att_freq = att_freq
        self.Q = Q
        self.observation_space = env.observation_space
        self.S_prev = None
        self.np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1]) 
        self.Q_obs_index_lims = [30, 30, 50, 50] #amount of discrete states for each dim of the observation
        print("Initialized pattanaik adv agent with att_freq: ", att_freq)

    #Transforms into discrete states so they can be used as indices of the Q-matrix dimensions (see self.obs_index_lims for max index sizes)
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
            S_actual, R, done, _ = self.env.step(A) #get the normal values of R and done

            #Will have to convert to a discrete state for Q matrix indices, then convert back to original continuous scale to return to the target agent. 
            S_actual = self.get_discrete_state(S_actual)

            #NOTe - change this to just an np.max
            a_optimum = np.argmax(self.Q[S_actual]) #Get optimum action from the current state
            # Q_optimum = Q[S_actual + (a_optimum,)]      #find Q value for this optimum action on actual state. Only needed for environments with greater than 2 actions

            #find neighbouring states by looping through forward and backward perturbations on each dimension.
            #running through the neighbouring states, we check what the optimum action that the policy would choose in this state is, then what Q value this produces when acted upon the current state. We then choose the false state that causes the action with the lowest Q value.
            #For this environment, this corresponds to choosing the first nieghbouring state that changes the action to the opposite one to the optimum.
            Snew = self.check_neighbours(S_actual,a_optimum)
            
            return self.get_continuous_state(Snew), R, done, _  
        else:
            return self.env.step(A)

    #Note that the return values are numpy arrays of dim 4.
    def check_neighbours(self, S, a_opt):
        #Perturbs each dim by -1,0 and +1 and gets all permutations to get all immediate neighbours.
        # S_neighbours = np.zeros(3*3*3*3,dtype=(int,4)) #array of correct size, containing tuples of 4 integeres
        # step = 0
        S_i = S[0]
        S_j = S[1]
        S_k = S[2]
        S_l = S[3]
        for i in [S_i-1, S_i, S_i+1]:
            for j in [S_j-1, S_j, S_j+1]:
                for k in [S_k-1, S_k, S_k+1]:
                    for l in [S_l-1, S_l, S_l+1]:
                            S_neighbour= (i,j,k,l)
                            a_opt_new = np.argmax(self.Q[S_neighbour])
                            if a_opt_new != a_opt:
                                return S_neighbour
                            # if self.Q[S_actual + (a_opt_new,)] < Q_optimum: #only needed for environments with greater than 2 actions
                            #     Snew = S_neighbour
                            #     break
        return S
        # return S_neighbours

    def reset(self):
        return self.env.reset()
    
    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

algtype = "QL"    #Either QL or SARSA works
alpha = 0.1       #step parameter
gamma = 0.95       #discount factor
epsilon = 1    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
epsilon_decay_value = 0.99995
n = 60000         #number of episodes

Q_file = open('cartpole/trained_Q.p', 'rb')
Q = pickle.load(Q_file)      #obtained a Q-matrix estimate from previous training 
Q_file.close()   

print("Using QL, huang attack on training, for different attack frequencies, versus the naive attack:")
#Set up the open ai gym environment twice
cartpole_env1 = gym.envs.make("CartPole-v1")
cartpole_env2 = gym.envs.make("CartPole-v1")

att_freq_arr =  [1,2,3,5,8,10,20,50] #Attack frequencies to try
plot_episodes = np.linspace(0, n, int(n/1000 + 1)) #Episodes at which we take a data point (every 1000)
fig, ax = plt.subplots() #set up graph
#Plot the results of the naive attack and the pattanaik attack for each attack frequency
for att_freq in att_freq_arr:
    #Adversarial man-in-the-middle agents and target agents to be attacked (who think they are interacting with the real cartpole environments directly)
    adv_env1 = cartpole_adv_pattanaik_training(cartpole_env1, att_freq, Q) 
    adv_env2 = naive_attack.cartpole_adv_randomstate_training(cartpole_env2,att_freq)
    target_agent1 = cp.target_cartpole_agent(adv_env1)
    target_agent2 = cp.target_cartpole_agent(adv_env2)

    #Peform training and plot
    plot_rewards1, _ = target_agent1.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n) 
    plot_rewards2, _ = target_agent2.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n)
    thislabel = 'Pattanaik, AF: ' + str(att_freq)
    ax.plot(plot_episodes,plot_rewards1, label=thislabel)
    thislabel = 'Naive, AF: ' + str(att_freq)
    ax.plot(plot_episodes,plot_rewards2, label=thislabel)

#Then train the base environment with no attacks as a benchmark
target_agent = cp.target_cartpole_agent(cartpole_env1) 
plot_rewards, _ = target_agent.train(algtype,alpha,gamma,epsilon,epsilon_decay_value,n)
thislabel = 'No attack'
ax.plot(plot_episodes,plot_rewards, label=thislabel)

ax.set_xlabel('Episode Number')
ax.set_ylabel('Average Reward Per 1000 Episodes')
ax.set_title('Average Reward During Training with QL on cart-pole Environment, Naive vs Pattanaik Attacks')
ax.legend() 
plt.show()
#Try to plot over the 60000 episodes, while changing the attack frequency to see the effect of the attack.                 

