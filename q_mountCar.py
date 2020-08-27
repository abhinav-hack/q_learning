# importing librarys

import gym
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 10000

SHOW_EVERY = 1000


env = gym.make("MountainCar-v0")            # making enviorment
#print("observation_space", env.observation_space)
#print("action_space", env.action_space)

discrete_os_size = [40]* len(env.observation_space.high)        # creating array of 20* len of observation 
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/discrete_os_size   # creating discrete win

epsilon = 0.0          # adding epsilon for randomness
start_epsilon = 1
end_epsilon = EPISODES//2
epsilon_decay_value = epsilon /(end_epsilon - start_epsilon)

q_table = np.load("/root/Documents/reinforcement/q_table/0-q_table.npy")
#q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size+[env.action_space.n])) # qtable 20*20*3

ep_rewards = []
aggr_ep_rewards = {"ep":[], "avg":[], "max":[], "min":[]}

#print("win size", discrete_os_win_size)
def get_discrete_state(state):      # Converting state to discrete states for input
    discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(int))


for lap in range(EPISODES):     # running over EPISODES

    episod_reward = 0
    if lap % SHOW_EVERY == 0:       # for rendering every 
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

        episod_reward += reward
        if render :                 #immediatly render after action for total video
            env.render()  

        
        new_discrete_state = get_discrete_state(new_state)      #get discrete value of new state
        
                                      #render image
        if not done:                                        
            max_future_q = np.max(q_table[new_discrete_state])      # taking value of max q of new state
            current_q = q_table[discrete_state+(action, )]          # taking value of max q of previous state
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)     # eq for new q
            q_table[discrete_state +(action, )] = new_q             #updating new q for current state
        elif new_state[0] >= env.goal_position:                 # if new state is win
            q_table[discrete_state + (action,)] = 0             # update previous q with max value 0
#            print("winner",lap, episod_reward)
        discrete_state = new_discrete_state
    
    # epsilon decay 
    if start_epsilon <= epsilon <= end_epsilon:
        epsilon -= epsilon_decay_value

    ep_rewards.append(episod_reward)

    if not lap % SHOW_EVERY:
        average_reward = sum(ep_rewards[-SHOW_EVERY:])/ len(ep_rewards[-SHOW_EVERY:])
        aggr_ep_rewards['ep'].append(lap)
        aggr_ep_rewards['avg'].append(average_reward)
        aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
        aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
        
        print(f"Episode:{lap} avg:{average_reward} min:{min(ep_rewards[-SHOW_EVERY:])} max:{max(ep_rewards[-SHOW_EVERY:])}")

    if lap % 200 ==0:
        np.save(f"/root/Documents/reinforcement/q_table/{lap}-q_table.npy", q_table)

env.close()

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label= 'avg')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label= 'min')
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label= 'max')
plt.legend(loc='best')
plt.show()


# adding extra comment to check git through vscode
