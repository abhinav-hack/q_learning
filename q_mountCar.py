

import gym
import numpy as np

learning_rate = 0.1
discount = 0.95
episodes = 1000

show_every = 100


env = gym.make("MountainCar-v0")
#print("observation_space", env.observation_space)
#print("action_space", env.action_space)

discrete_os_size = [20]* len(env.observation_space.high)
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/discrete_os_size
q_table = np.random.uniform(low=-2, high=0, size=(discrete_os_size+[env.action_space.n]))

#print("win size", discrete_os_win_size)
def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low)/ discrete_os_win_size
    return tuple(discrete_state.astype(int))


for lap in range(episodes):

    if lap % show_every == 0:
        render = True
    else :
        render = False

    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        action = np.argmax(q_table[discrete_state])
        new_state, reward, done, _ = env.step(action)
        if reward==0:
            print(reward)
        new_discrete_state = get_discrete_state(new_state)
        
        if render :
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state+(action, )]
            new_q = (1- learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state +(action, )] = new_q
        elif new_state[0] >= env.goal_position:
            q_table[discrete_state + (action,)] = 0
            print("winner",lap)
        discrete_state = new_discrete_state


env.close()


