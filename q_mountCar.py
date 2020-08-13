# importing librarys

import gym
import numpy as np

learning_rate = 0.1
discount = 0.95
episodes = 1000

show_every = 200


env = gym.make("MountainCar-v0")            # making enviorment
#print("observation_space", env.observation_space)
#print("action_space", env.action_space)

discrete_os_size = [20]* len(env.observation_space.high)        # creating array of 20* len of observation 
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/discrete_os_size   # creating discrete win

epsilon = 0.5           # adding epsilon for randomness
start_epsilon = 1
end_epsilon = episodes//2
epsilon_decay_value = epsilon /(end_epsilon - start_epsilon)


q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size+[env.action_space.n])) # qtable 20*20*3

#print("win size", discrete_os_win_size)
def get_discrete_state(state):      # Converting state to discrete states for input
    discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(int))


for lap in range(episodes):     # running over episodes

    if lap % show_every == 0:       # for rendering every 
        render = True
    else :
        render = False

    discrete_state = get_discrete_state(env.reset())    #initialize random states
    done = False
    while not done:

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])        #max value of qtable for that state 0,1,2
        else :
            action = np.random.randint(0, env.action_space.n)        #pick random value for that state 0,1,2

        
        new_state, reward, done, _ = env.step(action)       # use that action
        
        if render :                 #immediatly render after action for total video
            env.render()  

        if reward!=-1:                                      #  if reward is 0 that means flag is crossed
            print(reward)
        new_discrete_state = get_discrete_state(new_state)      #get discrete value of new state
        
                                      #render image
        if not done:                                        
            max_future_q = np.max(q_table[new_discrete_state])      # taking value of max q of new state
            current_q = q_table[discrete_state+(action, )]          # taking value of max q of previous state
            new_q = (1- learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)     # eq for new q
            q_table[discrete_state +(action, )] = new_q             #updating new q for current state
        elif new_state[0] >= env.goal_position:                 # if new state is win
            q_table[discrete_state + (action,)] = 0             # update previous q with max value 0
            print("winner",lap, q_table[discrete_state], reward)
        discrete_state = new_discrete_state
    
    # epsilon decay 
    if start_epsilon <= epsilon <= end_epsilon:
        epsilon -= epsilon_decay_value


env.close()

