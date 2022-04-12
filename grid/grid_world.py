import numpy as np
import random as rndm
import time 

#Single argu  ment to initialize for the type of algorithm. Either "QL" or "SARSA"
#If using Q-Learning, choice of epsilon is arbitrary but must be provided as an argument.
class target_grid_agent():
    def __init__(self, env):
        self.env = env
        self.observation_space = 16 #Number of states
        self.action_space = 4       #Number of actions
        self.iteration_max = 40     #Maximum number of iterations for training/testing. Stops infinite recursion.
    
    #Takes the arr from Q with one state and multiple actions, finds highest value and returns the index of the action column
    def choose_A(self, arr, isExploit):
        if isExploit:
            return np.argmax(arr)#indices of maximum Q value of the array
        else:
            return rndm.randint(0,3)

    #trains a policy based on the algtype algorithm given and other parameters
    def train(self, algtype, alpha, gamma, epsilon, n, fr_gap):
        if algtype not in {"QL", "SARSA"}:
            print("Invalid algorithm type, needs to be 'QL' or 'SARSA'. Default is 'QL'")
            self.algtype = "QL"
        else:
            self.algtype = type
        Q = np.random.uniform(low=0, high=1, size=((self.observation_space,) + (self.action_space,))) #Initialize the Q table
        plot_episodes = np.linspace(0, n, int(n/50 + 1)) #Data points to take results at (every 50 episodes)
        plot_rewards = np.zeros((int(n/50 + 1)))         #Stored rewards from these data points       
        t_total = 0  #For timing episodes and total training time
        R_total = 0  #For keeping track of total reward
        R_prior = 0  #For storing prev episode reward, for checking if we should change epsilon

        for i in range(n+1):
            t0 = time.time() #set initial time
            S = self.env.reset()          #choose random starting state
            Snew = S                      #initialise Snew
            done = False         #false until the terminal state is reached (S = 0)
            iteration = 1
            episode_R = 0

            #SARSA version uses epsilon greedy policy to update Q, Q-Learning uses the greedy policy
            #Epsilon=0 is pure exploitation, 1 is pure exploration.
            #Both algorithms use an epsilon greedy policy to select the current action but differ in how Q is updated
            isExploit = True
            while not done and iteration < self.iteration_max:
                isExploit = rndm.uniform(0,1) > epsilon  #epsilon-greedy
                Amax = self.choose_A(Q[S,:], isExploit)  #Get action for current move
                Snew, R, done = self.env.step(Amax)     #Find out the new state according to the action
                episode_R += R

                isExploit = not (self.algtype == "SARSA" and not isExploit)
                Anewmax = self.choose_A(Q[Snew,:], isExploit) #Get action to update Q with

                if(fr_gap != 0 and iteration % fr_gap == 0):
                    Q[S,Amax] = Q[S,Amax] + alpha*(rndm.randint(-10,-1) + gamma*Q[Snew,Anewmax] - Q[S,Amax]) #Update Q
                else:
                    Q[S,Amax] = Q[S,Amax] + alpha*(R + gamma*Q[Snew,Anewmax] - Q[S,Amax]) #Update Q
                S = Snew #Update agent's model of the state
                iteration += 1

            #Keep data for reward and print calculation time and average rewards
            t_ep = time.time() - t0
            t_total += t_ep
            R_total += episode_R
            R_prior = episode_R
            if i in plot_episodes: #every 50 episodes print the average time and the average reward
                time_average = t_total / 50
                print("Time Average: " + str(time_average))
                t_total = 0
                r_average = R_total / 50
                print("Reward Average: " + str(r_average))
                plot_rewards[int(i/50)] = r_average
                R_total = 0

        if i == n:
            #write final Q Table to test it later
            self.Q = Q
            # Q_file = open('grid_target_Q.p','wb')
            # pickle.dump(Q , Q_file)
            # Q_file.close()


        #Finding the optimum policy from Q table. This is formulated as a matrix, where each state has the optimum action in it
        # self.optimum_pi = np.zeros((16,1))
        # for i in range(1,16):
        #     isExploit = True
        #     A = self.choose_A(Q[i,:],isExploit)
        #     self.optimum_pi[i] = A
        # if(print_flag):
        #     print("optimum policy:\n {}".format(self.optimum_pi.reshape(4,4)))
        return plot_rewards, Q

    #For a given Snew and S, find the action that moves between the two.
    def find_A(self, S, Snew):
        diff = (Snew-S).astype(int)
        diff_mapping = {
        -4: 0,
         1: 1,
         4: 2,
        -1: 3
        }
        return diff_mapping.get(diff[0], 0) #return action 0 if not found as default


    #Does a n_test episodes and counts average reward for each starting square (with a ceiling cut-off for num of tries)
    #Then averages that reward over all squares for a single result.
    def test(self, n):
        #results is the total reward from episodes starting in that state, no_of_starts is the total times started in that state.
        print("Testing grid-world agent")
        results = np.zeros((self.observation_space,1))       
        no_of_starts = np.zeros((self.observation_space,1)) 
        
        for i in range(n):
            episode_R = 0              #accumulated reward for this episode
            start_S = self.env.reset() #choose random starting state
            Snew = S = start_S               
            no_of_starts[S] += 1
            done = False     #False until the terminal state is reached (S = 0)
            iteration = 1
            while not done and iteration < self.iteration_max:
                A = np.argmax(self.Q[S]) #use the Snew from the optimum policy matrix then find corresponding action from S to Snew
                Snew, R, done = self.env.step(A) #False is so it's not in optimum policy mode
                episode_R += R        #update reward for this step
                S = Snew          #update agent's state tracking
                iteration += 1
            results[start_S] += episode_R #add the total reward for this episode to the correct starting state location
        #divide accumulated R for each state by number of starts in each state
        results = np.divide(results, no_of_starts, out=np.zeros_like(results), where=no_of_starts!=0)
        print("Average reward matrix:\n ",results.reshape(4,4))
        print("Mean of matrix rewards: ",np.average(results))
        return results
        
    # #repeatedly trains (n_repeats times) using the algtype algorithm using the same parameters as train(), then tests them, then accumulates the average reward outputs and finds an overall average
    # def repeat_train_test(self, algtype, alpha, gamma, epsilon, n_train, fr_grap, n_repeats, n_test):
    #     results = np.zeros((16,1))
    #     for i in range(n_repeats):
    #         print("Completing run through", i)
    #         self.train(algtype, alpha, gamma, epsilon, n_train, fr_grap)
    #         results += self.test_opt_policy(n_test)
    #     results = results/n_repeats
    #     print("Average reward matrix:\n ",results.reshape(4,4))
    #     print("Mean of matrix rewards: ",np.average(results))

class grid_env:
    def __init__(self):
        #set up reward so if an action would take it off the board, make it stay still and lose 10 reward, otherwise reward
        # {F, , ,  },
        # {_,_,_,  },
        # { , ,_|, },
        # { , , ,  }
        self.observation_space = 16
        self.action_space = 4
        self.R = np.array([[-10, -1, -1,-10], [-10, -1, -1, -1], [-10, -1, -1, -1], [-10,-10, -1, -1],
                        [ -1, -1,-10,-10], [ -1, -1,-10, -1], [ -1, -1,-10, -1], [ -1,-10, -1, -1],
                        [-10, -1, -1,-10], [-10, -1, -1, -1], [-10,-10,-10, -1], [ -1,-10, -1,-10],
                        [ -1, -1,-10,-10], [ -1, -1,-10, -1], [-10, -1,-10, -1], [ -1,-10,-10, -1]])
        self.S = np.random.randint(0,self.observation_space)
    
    #takes input of action and returns new state according to board physics, alongside the reward of the current state
    #opt_pol_mode as True forces a deterministic change_state() to help figure out the optimum policy matrix.
    def step(self, A):
        #Don't move if at the board edge. Otherwise 0 up, 1 right, 2 down, 3 left
        doMove = rndm.randint(0,1) #50% chance of action actually changing state
        Snew = self.S                   #Returned new state
        if((Snew % 4 == 0 and A == 3) or (Snew < 4 and A == 0) or (Snew > 11 and A == 2) or (Snew % 4 == 3 and A == 1) or (doMove == 0)):
            pass 
        elif(A == 0):
            Snew = Snew-4
        elif(A == 1):
            Snew = Snew+1
        elif(A == 2):
            Snew = Snew+4
        elif(A == 3):
            Snew = Snew-1
        else:
            print('change_state() broken at state {} and action {}'.format(Snew,A))
        R = self.R[self.S,A]
        self.S = Snew           #keep track of state
        done = (Snew == 0)           #done if S is at state 0
        return Snew, R, done  #return state for next step and current reward

    #Starts an episode within the environment and returns the state.
    def reset(self):
        self.S = np.random.randint(1,self.observation_space)  #Generate random start state (1 to 15)
        return self.S