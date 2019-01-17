# Hyper Parameters
BATCH_SIZE = 24
LR = 0.01           # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
LAMBDA = 0.9                # parameter decay
TARGET_REPLACE_ITER = 24*7*4*2    # target update frequency (every two months)
MEMORY_CAPACITY = 24*7*4*6      # store upto six month worth of memory   

N_ACTIONS = 10 #no. of duty cycles (0,1,2,3,4)
N_STATES = 3 #number of state space parameter [batt, enp, henergy]