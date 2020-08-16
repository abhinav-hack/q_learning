import numpy as np
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
import time

style.use("ggplot")

SIZE = 10
HM_EPISODES = 2000
MOVE_PENALTY = 1 
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.0
EPS_DECAY = 0.9998
SHOW_EVERY = 200

start_q_table = "/root/Documents/reinforcement/qtable.pickel"   # add None for the first run

LEARNING_RATE = 0.1
DISCOUNT = 0.95

PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3

d = {1 : (255, 175, 0),
     2 : (0, 255, 0),
     3 : (0, 0, 255)}
     
class Blob:
    def __init__(self):
        self.x = np.random.randint(0, SIZE)
        self.y = np.random.randint(0, SIZE)
    
    def __str__(self):
        return f"{self.x}, {self.y}"

    def __sub__(self, other):
        return  (self.x - other.x, self.y - other.y)

    def action(self, choice):               # added act to take action with x,y =0 
        if choice == 0 :                                #only will able to  move veritcally and hori..
            self.move(x=1, y=0, act=True)
        if choice == 1 :
            self.move(x=-1, y=0, act=True)
        if choice == 2 :
            self.move(x=0, y=1, act=True)
        if choice == 3 : 
            self.move(x=0, y=-1, act=True)
    def move(self, x=False, y=False, act=False):        # act false will move any direction diagonally
        if act == False:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x

        if act == False:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        if self.x < 0 :
            self.x = 0
        elif self.x > SIZE-1:
            self.x =  SIZE-1

        if self.y < 0 :
            self.y = 0
        elif self.y > SIZE-1:
            self.y =  SIZE-1  

if start_q_table is None:
    q_table = {}
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1, y1), (x2, y2))]= [np.random.uniform(-5, 0) for i in range(4)]
    print("lenth of q_table dict",len(q_table))
else :
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
lose_count = 0
for episode in range(HM_EPISODES):
    player =  Blob()
    food = Blob()
    enemy = Blob()

    if episode % SHOW_EVERY == 0 :
        print(f"on # {episode}, epsilon: {epsilon}")
        print(f"{SHOW_EVERY} ep mean {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        if np.random.random() > epsilon:
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        
        player.action(action)
        ### 
        enemy.move()
        food.move()
        ###

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        new_obs = (player- food, player- enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]
        
        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        elif reward == -ENEMY_PENALTY:
            new_q = -ENEMY_PENALTY
        else:
            new_q = (1- LEARNING_RATE)* current_q + LEARNING_RATE*(reward+ DISCOUNT* max_future_q)

        q_table[obs][action] =new_q

        if show:
            env = np.zeros((SIZE, SIZE, 3), dtype= np.uint8)
            env[food.y][food.x] = d[FOOD_N]
            env[enemy.y][enemy.x] = d[ENEMY_N]
            env[player.y][player.x] = d[PLAYER_N]

            img = Image.fromarray(env, "RGB")
            img = img.resize((300, 300))
            cv.imshow("Game", np.array(img))
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            
            if reward == -ENEMY_PENALTY:
                lose_count += 1
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            break 
    
    

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/ SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"rewad {SHOW_EVERY} ma")
plt.xlabel("episode #")
plt.show()
cv.destroyAllWindows()

print("match lose:",lose_count)
with open(f"/root/Documents/reinforcement/qtable.pickel", "wb") as f:     # change path to your directory
    pickle.dump(q_table, f)
