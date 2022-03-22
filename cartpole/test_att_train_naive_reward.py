# ########### Adv attack on TRAINING by RANDOM NOISE, changing the REWARD recieved ##########
# class cartpole_adv_randomreward_training():
#     def __init__(self, env, att_freq):
#         self.env = env
#         self.action_space = env.action_space
#         self.step_counter = 0
#         self.att_freq = att_freq
#         self.observation_space = env.observation_space
#         print("Initialized random-state adv agent with att_freq: ", att_freq)

#     #Every step_counter iterations, it will step the environment but provide a random state back to the training agent.
#     #The random state only affects the cart position and the pole position for now. Speeds are set to 0
#     def step(self, A):
#         self.step_counter += 1
#         if self.step_counter % self.att_freq == 0:
#             Snew , R, done, _ = self.env.step(A) 
#             return Snew, 0, done, _  #Simply return 0 instead on every attack iteration.
#         else:
#             return self.env.step(A)

#     def reset(self):
#         return self.env.reset()
    
#     def render(self):
#         self.env.render()

#     def close(self):
#         self.env.close()
