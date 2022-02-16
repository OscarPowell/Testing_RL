import numpy as np
import random as rndm

#Base class with generic methods.
# class target_ML_agent():
#     def __init__(self, env):
#         self.env = env
    
#     #Used for changing the environment after initializing
#     def set_env(self, env):
#         self.env = env


#Single argument to initialize for the type of algorithm. Either "QL" or "SARSA"
#If using Q-Learning, choice of epsilon is arbitrary but must be provided as an argument.
class grid_4x4_ex():
    def __init__(self, env):
        self.env = env
    
    #Used for changing the environment after initializing
    def set_env(self, env):
        self.env = env
    
    #Takes the arr from Q with one state and multiple actions, finds highest value and returns the index of the action column
    def choose_A(self, arr, isExploit):
        if isExploit:
            indices = np.where(arr == np.amax(arr)) #indices of maximum Q value of the array
            return indices[0][0]
        else:
            return rndm.randint(0,3)

    #trains a policy based on the algtype algorithm given and other parameters
    def train(self, algtype, alpha, gamma, epsilon, n, fr_gap, print_flag=True):
        if algtype not in {"QL", "SARSA"}:
            print("Invalid algorithm type, needs to be 'QL' or 'SARSA'. Default is 'QL'")
            self.algtype = "QL"
        else:
            self.algtype = type
        Q = np.zeros((16,4))  #arbitrary initial Q Table estimate ((16 states, 4 actions) value function)
        
        for i in range(n):
            S = self.env.reset()          #choose random starting state
            Snew = S                      #initialise Snew
            done = False         #false until the terminal state is reached (S = 0)
            iteration = 1

            #SARSA version uses epsilon greedy policy to update Q and choose state, Q-Learning uses the greedy policy only to choose state.
            #Epsilon=0 is pure exploitation, 1 is pure exploration.
            #Both algorithms use an epsilon greedy policy to select the current action but differ in how Q is updated
            isExploit = True
            while not done and iteration < 40:
                isExploit = rndm.uniform(0,1) > epsilon  #epsilon-greedy
                Amax = self.choose_A(Q[S,:], isExploit)  #Get action for current move
                Snew, R, done = self.env.step(Amax)     #Find out the new state according to the action

                isExploit = not (self.algtype == "SARSA" and not isExploit)
                Anewmax = self.choose_A(Q[Snew,:], isExploit) #Get action to update Q with

                if(fr_gap != 0 and iteration % fr_gap == 0):
                    Q[S,Amax] = Q[S,Amax] + alpha*(rndm.randint(-10,-1) + gamma*Q[Snew,Anewmax] - Q[S,Amax]) #Update Q
                else:
                    Q[S,Amax] = Q[S,Amax] + alpha*(R + gamma*Q[Snew,Anewmax] - Q[S,Amax]) #Update Q
                S = Snew #Update agent's model of the state
                iteration += 1

        #Finding the optimum policy from Q table. This is formulated as a matrix, where each state has the optimum action in it
        self.optimum_pi = np.zeros((16,1))
        for i in range(1,16):
            isExploit = True
            A = self.choose_A(Q[i,:],isExploit)
            self.optimum_pi[i] = A
        if(print_flag):
            print("optimum policy:\n {}".format(self.optimum_pi.reshape(4,4)))
        return self.optimum_pi

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
    def test_opt_policy(self, n_test, print_flag=True):
        #results is the total reward from episodes starting in that state, no_of_starts is the total times started in that state.
        results = np.zeros((16,1))       
        no_of_starts = np.zeros((16,1))
        
        for i in range(n_test):
            acc_R = 0                   #accumulated reward for this episode
            start_S = self.env.reset()  #choose random starting state
            Snew = S = start_S               
            no_of_starts[S] += 1
            done = False     #False until the terminal state is reached (S = 0)
            iteration = 1
            while not done and iteration < 40:
                A = self.optimum_pi[S].astype(int) #use the Snew from the optimum policy matrix then find corresponding action from S to Snew
                Snew, R, done = self.env.step(A) 
                acc_R += R       
                S = Snew          
                iteration += 1
            results[start_S] += acc_R #add the total reward for this episode to the correct starting state location
        #divide accumulated R for each state by number of starts in each state
        results = np.divide(results, no_of_starts, out=np.zeros_like(results), where=no_of_starts!=0)
        if print_flag:
            print("Average reward matrix:\n ",results.reshape(4,4))
            print("Mean of matrix rewards: ",np.average(results))
        return results
        
    #repeatedly trains (n_repeats times) using the algtype algorithm using the same parameters as train(), then tests them, then accumulates the average reward outputs and finds an overall average
    def repeat_train_test(self, algtype, alpha, gamma, epsilon, n_train, fr_grap, n_repeats, n_test, print_flag=True):
        results = np.zeros((16,1))
        for i in range(n_repeats):
            print("Completing run through", i)
            self.train(algtype, alpha, gamma, epsilon, n_train, fr_grap, print_flag)
            results += self.test_opt_policy(n_test, print_flag)
        results = results/n_repeats
        print("Average reward matrix:\n ",results.reshape(4,4))
        print("Mean of matrix rewards: ",np.average(results))
        return np.average(results)
    
class grid_env:
    def __init__(self):
        #set up reward so if an action would take it off the board, make it stay still and lose 10 reward, otherwise reward
        # {F, , ,  },
        # {_,_,_,  },
        # { , ,_|, },
        # { , , ,  }
        self.R = np.array([[-10, -1, -1,-10], [-10, -1, -1, -1], [-10, -1, -1, -1], [-10,-10, -1, -1],
                        [ -1, -1,-10,-10], [ -1, -1,-10, -1], [ -1, -1,-10, -1], [ -1,-10, -1, -1],
                        [-10, -1, -1,-10], [-10, -1, -1, -1], [-10,-10,-10, -1], [ -1,-10, -1,-10],
                        [ -1, -1,-10,-10], [ -1, -1,-10, -1], [-10, -1,-10, -1], [ -1,-10,-10, -1]])
        self.S = np.random.randint(0,16)
    
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
        self.S = np.random.randint(1,16)  #Generate random start state (1 to 15)
        return self.S

class rand_noise_adv_env:
    def __init__(self, att_gap):
        #set up reward so if an action would take it off the board, make it stay still and lose 10 reward, otherwise reward
        # {F, , ,  },
        # {_,_,_,  },
        # { , ,_|, },
        # { , , ,  }
        self.R = np.array([[-10, -1, -1,-10], [-10, -1, -1, -1], [-10, -1, -1, -1], [-10,-10, -1, -1],
                            [ -1, -1,-10,-10], [ -1, -1,-10, -1], [ -1, -1,-10, -1], [ -1,-10, -1, -1],
                            [-10, -1, -1,-10], [-10, -1, -1, -1], [-10,-10,-10, -1], [ -1,-10, -1,-10],
                            [ -1, -1,-10,-10], [ -1, -1,-10, -1], [-10, -1,-10, -1], [ -1,-10,-10, -1]])
        self.S = np.random.randint(0,16)
        self.att_gap = att_gap
    
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
        #After keeping track of the actual state, it provides fake state data to persuade the agent algorithm to give a suboptimal action
        if(self.counter % self.att_gap == 0):
            Snew = rndm.randint(1,15)
        self.counter += 1
        return Snew, R, done  #return state for next step and current reward

    #Starts an episode within the environment and returns the state.
    def reset(self):
        self.S = rndm.randint(1,15)  #Generate random start state (1 to 15)
        self.counter = 1                  #initializes a counter
        return self.S

#This agent is set up to choose the action with the lowest adjacent reward, each time it attacks.
#Is formulated as a black box with no access to the Q function.
class min_reward_agent_env:
    def __init__(self, att_gap):
        #set up reward so if an action would take it off the board, make it stay still and lose 10 reward, otherwise reward
        # {F, , ,  },
        # {_,_,_,  },
        # { , ,_|, },
        # { , , ,  }
        self.R = np.array([[-10, -1, -1,-10], [-10, -1, -1, -1], [-10, -1, -1, -1], [-10,-10, -1, -1],
                            [ -1, -1,-10,-10], [ -1, -1,-10, -1], [ -1, -1,-10, -1], [ -1,-10, -1, -1],
                            [-10, -1, -1,-10], [-10, -1, -1, -1], [-10,-10,-10, -1], [ -1,-10, -1,-10],
                            [ -1, -1,-10,-10], [ -1, -1,-10, -1], [-10, -1,-10, -1], [ -1,-10,-10, -1]])
        self.S = np.random.randint(0,16)
        self.att_gap = att_gap
    
    #takes input of action and returns new state according to board physics, alongside the reward of the current state
    #opt_pol_mode as True forces a deterministic change_state() to help figure out the optimum policy matrix.
    def step(self, A):
        #Don't move if at the board edge. Otherwise 0 up, 1 right, 2 down, 3 left
        doMove = rndm.randint(0,1) #50% chance of action actually changing state
        R = self.R[self.S,A]

        Snew = self.find_new_state(self.S, A, doMove)
        self.S = Snew           #keep track of state
        done = (Snew == 0)           #done if S is at state 0
        #After keeping track of the actual state, it provides fake state data to persuade the agent algorithm to give a suboptimal action
        if(self.counter % self.att_gap == 0):
            A_min = np.argmin(self.R[self.S,:])
            Snew = self.find_new_state(self.S, A_min, True) # note it chooses the first minimum value
        self.counter += 1 
        return Snew, R, done  #return state for next step and current reward

    def find_new_state(self,S,A,doMove):
        if((S % 4 == 0 and A == 3) or (S < 4 and A == 0) or (S > 11 and A == 2) or (S % 4 == 3 and A == 1) or (doMove == 0)):
            Snew = S 
        elif(A == 0):
            Snew = S-4
        elif(A == 1):
            Snew = S+1
        elif(A == 2):
            Snew = S+4
        elif(A == 3):
            Snew = S-1
        else:
            Snew = S
            print('change_state() broken at state {} and action {}'.format(S,A))
        return Snew


    #Starts an episode within the environment and returns the state.
    def reset(self):
        self.S = rndm.randint(1,15)  #Generate random start state (1 to 15)
        self.counter = 1                  #initializes a counter
        return self.S
