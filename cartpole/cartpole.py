import numpy as np
import random as rndm
import gym
import math
import time 
import matplotlib.pyplot as plt
import pickle

# State space: cart position, cart velocity, pole angle, pole tip velocity
# For now just set the environment as the cartpole, but when it works, change this environment to an adversarial one.
class target_cartpole_agent():
    def __init__(self, env):
        self.env = env
        # self.env = gym.envs.make("CartPole-v1") #use if env not working
        print("Initialising cartpole training agent")
        self.np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1]) #steps for each.

    #Used for changing the environment after initializing - can be used later to set up adversarial attack
    def set_env(self, env):
        self.env = env
    
    #Takes the arr from Q with one state and multiple actions, finds highest value and returns the index of the action column
    def choose_A(self, arr, isExploit):
        if isExploit:
            action = np.argmax(arr) #take greedy action (highest reward)
        else:
            # action = np.random.randint(0, self.env.action_space.n) #take random action
            action = self.env.action_space.sample()
        return action

    #turns the tuple state into a tuple containing the discrete state.
    def get_discrete_state(self, state):
        discrete_state = state/(self.np_array_win_size)+ np.array([15,10,1,10])
        return tuple(discrete_state.astype(int))

    #trains a policy based on the algtype algorithm given and other parameters
    def train(self, algtype, alpha, gamma, epsilon, epsilon_decay_value, n):
        if algtype not in {"QL", "SARSA"}:
            print("Invalid algorithm type, needs to be 'QL' or 'SARSA'. Default is 'QL'")
            self.algtype = "QL"
        else:
            self.algtype = algtype

        
        plot_episodes = np.linspace(0, n, int(n/1000 + 1)) #every 1000 episodes we take a data point
        plot_rewards = np.zeros((int(n/1000 + 1)))
        
        Observation = [30, 30, 50, 50] #first two variables not as important as the other two.
        Q = np.random.uniform(low=0, high=1, size=(Observation + [self.env.action_space.n])) # Concatenates observation and action space to specify the dim of the Q-table (very large)        
        t_total = 0  #For timing episodes and total training time
        R_total = 0  #For keeping track of total reward
        R_prior = 0  #For storing prev episode reward, for checking if we should change epsilon

        for i in range(n+1):
            t0 = time.time() #set initial time
            S = self.get_discrete_state(self.env.reset()) #reset environment and get starting discrete state
            episode_R = 0      #initialise current episode reward
            Snew = S           #initialise Snew
            done = False       #false until the terminal state is reached (S = 0)

            if i in plot_episodes: #Keep track of episodes while it runs so we know it's working
                print("Episode: " + str(i))

            #SARSA version uses epsilon greedy policy to update Q and choose state, Q-Learning uses the greedy policy only to choose state.
            #Epsilon=0 is pure exploitation, 1 is pure exploration.
            #Both algorithms use an epsilon greedy policy to select the current action but differ in how Q is updated
            isExploit = False
            while not done:
                isExploit = rndm.uniform(0 ,1) > epsilon  #epsilon-greedy
                Amax = self.choose_A(Q[S], isExploit)    #Get action for current move
                Snew, R, done, _ = self.env.step(Amax)   #Find out the new continuous state according to the action
                Snew = self.get_discrete_state(Snew)            #Find discrete version of state
                episode_R += R

                if i % 2000 == 0:
                    self.env.render()
                
                if i == n:
                    #write final Q Table to test it later
                    self.Q = Q
                    # Q_file = open('pattanaik_target_Q.p','wb')
                    # pickle.dump(Q , Q_file)
                    # Q_file.close()

                if not done:
                    if isExploit or self.algtype == 'QL':
                        Q_Snew = np.max(Q[Snew])
                    else:
                        Q_Snew = rndm.choice(Q[Snew])

                    Q[S + (Amax,)] = (1-alpha)*Q[S + (Amax,)] + alpha*(R + gamma * Q_Snew) #Update Q
                    S = Snew #Update agent's model of the state

            #Now we change the epsilon value to allow more exploitation later on and more exploration earlier on
            if epsilon > 0.05: #epsilon modification
                if episode_R > R_prior and i > 10000:
                    epsilon = math.pow(epsilon_decay_value, i - 10000)

                    if i % 500 == 0:
                        print("Epsilon: " + str(epsilon))            

            t_ep = time.time() - t0
            t_total += t_ep
            R_total += episode_R
            R_prior = episode_R
            if i in plot_episodes: #every 1000 episodes print the average time and the average reward
                time_average = t_total / 1000
                print("Time Average: " + str(time_average))
                t_total = 0
                r_average = R_total / 1000
                print("Reward Average: " + str(r_average))
                plot_rewards[int(i/1000)] = r_average
                R_total = 0

        self.env.close()
        return plot_rewards, Q

    #Tests the policy, which is defined by the Q matrix' optimum values. Note train() has to have already have been called.
    #n is the number of runs to test. The return value is the average reward per episode for the testing.
    def test(self,n):
        total_R = 0
        for i in range(n):
                S = self.get_discrete_state(self.env.reset()) #reset environment and get starting discrete state
                done = False
                while not done:
                    Amax = np.argmax(self.Q[S])
                    S, R, done, _ = self.env.step(Amax)   #Find out the new continuous state according to the action
                    S = self.get_discrete_state(S)            #Find discrete version of state
                    total_R += R
        return total_R/n