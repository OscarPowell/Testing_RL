#Function to replace the target_cartpole_agent, so we may change the training algorithm slightly
import cartpole as cp
import numpy as np
import random as rndm
import math

class cartpole_epsilon_change(cp.target_cartpole_agent):
    def __init__(self, env):
        super().__init__(env)

    #trains a policy based on the algtype algorithm given and other parameters
    #Adds the parameter exploit wait which was previously set at 10,000. Iterations before epsilon starts to decrease.
    def train(self, algtype, alpha, gamma, epsilon, epsilon_decay_value, exploit_wait, n):
        if algtype not in {"QL", "SARSA"}:
            print("Invalid algorithm type, needs to be 'QL' or 'SARSA'. Default is 'QL'")
            self.algtype = "QL"
        else:
            self.algtype = algtype
        
        plot_episodes = np.linspace(0, n, int(n/1000 + 1)) #every 1000 episodes we take a data point
        plot_rewards = np.zeros((int(n/1000 + 1)))
        
        Observation = [30, 30, 50, 50] #first two variables not as important as the other two.
        Q = np.random.uniform(low=0, high=1, size=(Observation + [self.env.action_space.n])) # Concatenates observation and action space to specify the dim of the Q-table (very large)        
        R_total = 0  #For keeping track of total reward
        R_prior = 0  #For storing prev episode reward, for checking if we should change epsilon

        for i in range(n+1):
            S = self.get_discrete_state(self.env.reset()) #reset environment and get starting discrete state
            episode_R = 0      #initialise current episode reward
            Snew = S           #initialise Snew
            done = False       #false until the terminal state is reached (S = 0)

            # if i in plot_episodes: #Keep track of episodes while it runs so we know it's working
            #     print("Episode: " + str(i))

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

                # if i % 2000 == 0:
                #     self.env.render()
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
            if epsilon > 0.05 and episode_R > R_prior and i > exploit_wait:
                epsilon = math.pow(epsilon_decay_value, i - exploit_wait)

                # if i % 500 == 0:
                #     print("Epsilon: " + str(epsilon))       

            R_total += episode_R
            R_prior = episode_R
            if i % 1000 == 0: #every 1000 episodes print the average time and the average reward
                r_average = R_total / 1000
                # print("Reward Average: " + str(r_average))
                plot_rewards[int(i/1000)] = r_average
                R_total = 0

        self.env.close()
        return plot_rewards, Q

