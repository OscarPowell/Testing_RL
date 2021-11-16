import numpy as np

#Takes the arr from Q with one state and multiple actions, finds highest value and returns the index of the action column
def get_max_A(arr):
    indices = np.where(arr == np.amax(arr)) #indices of maximum Q value of the array
    return indices[0][0]

#takes input of action and returns new state according to board physics
def change_state(S,A):
    #Don't move if at the board edge. Otherwise 0 up, 1 right, 2 down, 3 left
    if((S % 3 == 0 and A == 3) or (S < 3 and A == 0) or (S > 5 and A == 2) or (S % 3 == 2 and A == 1)):
        return S
    elif(A == 0):
        return S-3
    elif(A == 1):
        return S+1
    elif(A == 2):
        return S+3
    elif(A == 3):
        return S-1
    else:
        print('change_state() broken at state {} and action {}'.format(S,A))
        return S

alpha, gamma, n = 0.5, 0.9, 200    #step parameter, discount factor, number of episodes

Q = np.zeros((9,4))                #arbitrary initial Q Table estimate ((9 states, 4 actions) value function)

#set up reward so if an action would take it off the board, make it stay still and lose 10 reward, otherwise reward=-1
R = np.array([[-10, -1, -1, -10],
              [-10, -1, -1,  -1],
              [-10,-10, -1,  -1],
              [ -1, -1, -1, -10],
              [ -1, -1, -1,  -1],
              [ -1,-10, -1,  -1],
              [ -1, -1,-10, -10],
              [ -1, -1,-10,  -1],
              [ -1,-10,-10,  -1]])
S0_arr = np.random.randint(0,9,n)  #Generate starting states (1,to 9)

for i in range(n):
    S = S0_arr[i]          #choose random starting state
    Snew = S
    running = True         #true until the terminal state is reached (S = 0 or 8)
    # iteration = 1
    while running:
        if S == 0 or S == 8:
            running = False
        else:
            Amax = get_max_A(Q[S,:])       #Get action that maximizes current move
            Snew = change_state(S,Amax)      #Find out the new state according to the action
            Anewmax = get_max_A(Q[Snew,:]) #Get action that maximizes the next move
            Q[S,Amax] = Q[S,Amax] + alpha*(R[S,Amax]+ gamma*Q[Snew,Anewmax] - Q[S,Amax]) #Update Q
            S = Snew                         #Update state
            # print('Iteration {} at state {}'.format(iteration,S)) #debugging iteration check
            # iteration += 1


print('Final Q: {}'.format(Q))
