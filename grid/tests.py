import QL_SARSA as algs
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

############## TESTING THE RAW ALGORITHM WORKS #####################
# print("### TESTING THE RAW ALGORITHM WORKS ###")
# algtype = "SARSA"    #Either QL or SARSA works
# alpha = 0.5       #step parameter
# gamma = 0.9       #discount factor
# epsilon = 0.05    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
# n = 1600         #number of episodes 
# fr_gap = 0        #gap between iterations for false rewards provided (0 for no false rewards)
# n_repeats = 10
# print_flag = False
# n_test = 1600

# test_env = algs.grid_4x4_ex(algs.grid_env())
# print("Using SARSA:")
# result = test_env.repeat_train_test(algtype,alpha,gamma,epsilon,n,fr_gap,n_repeats,n_test,print_flag)

# algtype = "QL"
# print("Using QL:")
# result = test_env.repeat_train_test(algtype,alpha,gamma,epsilon,n,fr_gap,n_repeats,n_test,print_flag)

################## TESTING rand_noise_adv_env successfully changes state during TESTING #################################

# print("###TESTING rand_noise_adv_env successfully changes state during TESTING ###")

# #First train the target agent and test it on an ordinary environment:
# print("Training target agent:")
# algtype = "QL"    #Either QL or SARSA works
# alpha = 0.5       #step parameter
# gamma = 0.9       #discount factor
# epsilon = 0.05    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
# n = 1600          #number of episodes 
# fr_gap = 0        #gap between iterations for false rewards provided (0 for no false rewards)
# print_flag = True
# n_test = 1600
# target_agent = algs.grid_4x4_ex(algs.grid_env())
# target_agent.train(algtype, alpha, gamma, epsilon, n, fr_gap, print_flag)
# print("\nTesting the target agent on an ordinary environment:")
# _suppressed_output = target_agent.test_opt_policy(n_test, print_flag)

# #Then attack during testing of the trained policy with the adversarial agent acting as a man-in-the-middle
# #Every att_gap iterations an attack occurs
# att_gap_range = np.arange(1,8)
# results = np.zeros((7,1))
# for att_gap in att_gap_range:
#     print("\nTesting attack with gap", att_gap)
#     target_agent.set_env(algs.rand_noise_adv_env(att_gap))
#     results[int(att_gap)-1] = np.average(target_agent.test_opt_policy(n_test, print_flag))

# plt.plot(att_gap_range,results)
# plt.title("Effect of Random Noise (State Data, Blackbox) Adversarial Attack on 4x4 Grid ")
# plt.xlabel("Iteration Gap Between Attacks")
# plt.ylabel("Average Accumulated Reward")
# plt.show()

########### TESTING rand_noise_adv_env successfully changes state during TRAINING ##########
print("###TESTING rand_noise_adv_env successfully changes state during TRAINING ###")

print("Training target agent with adversarial man-in-the-middle:")
algtype = "SARSA"    #Either QL or SARSA works
alpha = 0.5       #step parameter
gamma = 0.9       #discount factor
epsilon = 0.05    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
n = 1600          #number of episodes 
fr_gap = 0        #gap between iterations for false rewards provided (0 for no false rewards)
print_flag = True
n_test = 1600

att_gap_range = np.arange(1,8)
results = np.zeros((7,1))
target_agent = algs.grid_4x4_ex(algs.grid_env())
for att_gap in att_gap_range:
    print("\nUsing attack with gap", att_gap)
    target_agent.set_env(algs.rand_noise_adv_env(att_gap))
    target_agent.train(algtype, alpha, gamma, epsilon, n, fr_gap, print_flag) # #First train the target agent while attacking it:
    target_agent.set_env(algs.grid_env()) #Set back to the normal environment for testing
    results[int(att_gap)-1] = np.average(target_agent.test_opt_policy(n_test, print_flag)) # See the average results of the training

plt.plot(att_gap_range,results)
plt.title("Effect of Random Noise (State Data) Attack During Training on 4x4 Grid ")
plt.xlabel("Iteration Gap Between Attacks")
plt.ylabel("Average Accumulated Reward")
plt.show()

########### TESTING min_reward_agent_env successfully changes state during TRAINING ##########
# print("###TESTING min_reward_agent_env successfully changes state during TRAINING ###")

# print("Training target agent with adversarial man-in-the-middle:")
# algtype = "QL"    #Either QL or SARSA works
# alpha = 0.5       #step parameter
# gamma = 0.9       #discount factor
# epsilon = 0.05    #exploration parameter (0 for pure exploitation, 1 for pure exploration)
# n = 1600          #number of episodes 
# fr_gap = 0        #gap between iterations for false rewards provided (0 for no false rewards)
# print_flag = True
# n_test = 1600

# att_gap_range = np.arange(1,8)
# results = np.zeros((7,1))
# target_agent = algs.grid_4x4_ex(algs.grid_env())
# for att_gap in att_gap_range:
#     print("\nUsing attack with gap", att_gap)
#     target_agent.set_env(algs.min_reward_agent_env(att_gap))
#     target_agent.train(algtype, alpha, gamma, epsilon, n, fr_gap, print_flag) # #First train the target agent while attacking it:
#     target_agent.set_env(algs.grid_env()) #Set back to the normal environment for testing
#     results[int(att_gap)-1] = np.average(target_agent.test_opt_policy(n_test, print_flag)) # See the average results of the training

# plt.plot(att_gap_range,results)
# plt.title("Effect of Random Noise (State Data) Attack During Training on 4x4 Grid ")
# plt.xlabel("Iteration Gap Between Attacks")
# plt.ylabel("Average Accumulated Reward")
# plt.show()