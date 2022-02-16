import numpy as np
import random as rndm
import gym
import math
import time

#State space: cart position, cart velocity, pole angle, pole tip velocity
env = gym.make("CartPole-v1")
print(env.action_space.n)

LEARNING_RATE = 0.1

DISCOUNT = 0.95
EPISODES = 60000
total = 0
total_reward = 0
prior_reward = 0

Observation = [30, 30, 50, 50] #first two variables not as important as the other two.
np_array_win_size = np.array([0.25, 0.25, 0.01, 0.1]) #steps for each.

epsilon = 1

epsilon_decay_value = 0.99995

q_table = np.random.uniform(low=0, high=1, size=(Observation + [env.action_space.n])) # Concatenates observation and action space to specify the dim of the Q-table (very large)
q_table.shape

def get_discrete_state(state):
    discrete_state = state/np_array_win_size+ np.array([15,10,1,10])
    return tuple(discrete_state.astype(np.int))

for episode in range(EPISODES + 1): #go through the episodes
    t0 = time.time() #set the initial time
    discrete_state = get_discrete_state(env.reset()) #get the discrete start for the restarted environment 
    done = False
    episode_reward = 0 #reward starts as 0 for each episode

    if episode % 2000 == 0: 
        print("Episode: " + str(episode))

    while not done: 

        if np.random.random() > epsilon:

            action = np.argmax(q_table[discrete_state]) #take cordinated action
        else:

            action = np.random.randint(0, env.action_space.n) #do a random ation

        new_state, reward, done, _ = env.step(action) #step action to get new states, reward, and the "done" status.

        episode_reward += reward #add the reward

        new_discrete_state = get_discrete_state(new_state)

        if episode % 2000 == 0: #render
            env.render()

        if not done: #update q-table
            max_future_q = np.max(q_table[new_discrete_state])

            current_q = q_table[discrete_state + (action,)]

            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

            q_table[discrete_state + (action,)] = new_q

        discrete_state = new_discrete_state

    if epsilon > 0.05: #epsilon modification
        if episode_reward > prior_reward and episode > 10000:
            epsilon = math.pow(epsilon_decay_value, episode - 10000)

            if episode % 500 == 0:
                print("Epsilon: " + str(epsilon))

    t1 = time.time() #episode has finished
    episode_total = t1 - t0 #episode total time
    total = total + episode_total

    total_reward += episode_reward #episode total reward
    prior_reward = episode_reward

    if episode % 1000 == 0: #every 1000 episodes print the average time and the average reward
        mean = total / 1000
        print("Time Average: " + str(mean))
        total = 0

        mean_reward = total_reward / 1000
        print("Mean Reward: " + str(mean_reward))
        total_reward = 0

env.close()    

#State space: cart position, cart velocity, pole angle, pole tip velocity
# class target_cartpole_agent():
#     def __init__(self, env):
#         self.env = gym.envs.make("CartPole-v1")
#         print()

    
#     #Used for changing the environment after initializing
#     def set_env(self, env):
#         self.env = env
    
#     #Takes the arr from Q with one state and multiple actions, finds highest value and returns the index of the action column
#     def choose_A(self, arr, isExploit):
#         if isExploit:
#             indices = np.where(arr == np.amax(arr)) #indices of maximum Q value of the array
#             return indices[0][0]
#         else:
#             return rndm.randint(0,3)

#     #trains a policy based on the algtype algorithm given and other parameters
#     def train(self, algtype, alpha, gamma, epsilon, n, fr_gap, print_flag=True):
#         if algtype not in {"QL", "SARSA"}:
#             print("Invalid algorithm type, needs to be 'QL' or 'SARSA'. Default is 'QL'")
#             self.algtype = "QL"
#         else:
#             self.algtype = type
#         Q = np.zeros((16,4))  #arbitrary initial Q Table estimate ((16 states, 4 actions) value function)
        
#         for i in range(n):
#             S = self.env.reset()          #choose random starting state
#             Snew = S                      #initialise Snew
#             done = False         #false until the terminal state is reached (S = 0)
#             iteration = 1

#             #SARSA version uses epsilon greedy policy to update Q and choose state, Q-Learning uses the greedy policy only to choose state.
#             #Epsilon=0 is pure exploitation, 1 is pure exploration.
#             #Both algorithms use an epsilon greedy policy to select the current action but differ in how Q is updated
#             isExploit = True
#             while not done and iteration < 40:
#                 isExploit = rndm.uniform(0,1) > epsilon  #epsilon-greedy
#                 Amax = self.choose_A(Q[S,:], isExploit)  #Get action for current move
#                 Snew, R, done = self.env.step(Amax)     #Find out the new state according to the action

#                 isExploit = not (self.algtype == "SARSA" and not isExploit)
#                 Anewmax = self.choose_A(Q[Snew,:], isExploit) #Get action to update Q with

#                 if(fr_gap != 0 and iteration % fr_gap == 0):
#                     Q[S,Amax] = Q[S,Amax] + alpha*(rndm.randint(-10,-1) + gamma*Q[Snew,Anewmax] - Q[S,Amax]) #Update Q
#                 else:
#                     Q[S,Amax] = Q[S,Amax] + alpha*(R + gamma*Q[Snew,Anewmax] - Q[S,Amax]) #Update Q
#                 S = Snew #Update agent's model of the state
#                 iteration += 1

#         #Finding the optimum policy from Q table. This is formulated as a matrix, where each state has the optimum action in it
#         self.optimum_pi = np.zeros((16,1))
#         for i in range(1,16):
#             isExploit = True
#             A = self.choose_A(Q[i,:],isExploit)
#             self.optimum_pi[i] = A
#         if(print_flag):
#             print("optimum policy:\n {}".format(self.optimum_pi.reshape(4,4)))
#         return self.optimum_pi

#     #For a given Snew and S, find the action that moves between the two.
#     def find_A(self, S, Snew):
#         diff = (Snew-S).astype(int)
#         diff_mapping = {
#         -4: 0,
#          1: 1,
#          4: 2,
#         -1: 3
#         }
#         return diff_mapping.get(diff[0], 0) #return action 0 if not found as default


#     #Does a n_test episodes and counts average reward for each starting square (with a ceiling cut-off for num of tries)
#     #Then averages that reward over all squares for a single result.
#     def test_opt_policy(self, n_test, print_flag=True):
#         #results is the total reward from episodes starting in that state, no_of_starts is the total times started in that state.
#         results = np.zeros((16,1))       
#         no_of_starts = np.zeros((16,1))
        
#         for i in range(n_test):
#             acc_R = 0                   #accumulated reward for this episode
#             start_S = self.env.reset()  #choose random starting state
#             Snew = S = start_S               
#             no_of_starts[S] += 1
#             done = False     #False until the terminal state is reached (S = 0)
#             iteration = 1
#             while not done and iteration < 40:
#                 A = self.optimum_pi[S].astype(int) #use the Snew from the optimum policy matrix then find corresponding action from S to Snew
#                 Snew, R, done = self.env.step(A) 
#                 acc_R += R       
#                 S = Snew          
#                 iteration += 1
#             results[start_S] += acc_R #add the total reward for this episode to the correct starting state location
#         #divide accumulated R for each state by number of starts in each state
#         results = np.divide(results, no_of_starts, out=np.zeros_like(results), where=no_of_starts!=0)
#         if print_flag:
#             print("Average reward matrix:\n ",results.reshape(4,4))
#             print("Mean of matrix rewards: ",np.average(results))
#         return results
        
#     #repeatedly trains (n_repeats times) using the algtype algorithm using the same parameters as train(), then tests them, then accumulates the average reward outputs and finds an overall average
#     def repeat_train_test(self, algtype, alpha, gamma, epsilon, n_train, fr_grap, n_repeats, n_test, print_flag=True):
#         results = np.zeros((16,1))
#         for i in range(n_repeats):
#             print("Completing run through", i)
#             self.train(algtype, alpha, gamma, epsilon, n_train, fr_grap, print_flag)
#             results += self.test_opt_policy(n_test, print_flag)
#         results = results/n_repeats
#         print("Average reward matrix:\n ",results.reshape(4,4))
#         print("Mean of matrix rewards: ",np.average(results))
#         return np.average(results)

