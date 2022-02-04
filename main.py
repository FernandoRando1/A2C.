import gym
import pyglet
import matplotlib.pyplot as plt
import numpy as np
import random
plt.style.use('_mpl-gallery')

import torch
import funct_and_nn as funct
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')


import statistics

# loads envirorment
env = gym.make('CartPole-v1')
number_of_episode = 5000
max_time = 5000

resulting_time_ls = []

# set q earning values
alpha = .5
gamma = .9


# init networks
#net = funct.NeuralNetwork(4,20,2,20,1).to(device)
policy_net = funct.policy_net_(4,50,2)
value_net = funct.value_net_(4,100,1)

# set net varibles
#learning_rate = .005
policy_learning_rate = .005
value_learning_rate = .0005


##optimizer_net = torch.optim.SGD(net.parameters(), lr=learning_rate)
#optimizer_net = torch.optim.Adam(net.parameters(), lr=learning_rate)

policy_opt = torch.optim.SGD(policy_net.parameters(), lr=policy_learning_rate)
value_opt = torch.optim.Adam(value_net.parameters(), lr=value_learning_rate)

# temp
# define action list
action_ls_ = [0, 1]


for i_episode in range(number_of_episode):
    total_reward = 0

    observation = env.reset()

    pre_state = observation
    post_state = observation

    is_term_ls = []
    pre_state_ls = []
    post_state_ls = []
    action_ls = []
    reward_ls = []



    for t in range(max_time):   # game loop

        #env.render()  # loads game window not needed

        # get pre state
        pre_state = observation

        # policy
        pre_state_tensor_ = torch.FloatTensor(pre_state)
        #action = funct.policy(pre_state_tensor_,action_ls_,net)
        action = funct.policy(pre_state_tensor_, action_ls_, policy_net)


        # update envirorment
        observation, reward, done, info = env.step(action)

        post_state = observation

        total_reward += reward

        pre_state_ls.append(pre_state)
        post_state_ls.append(post_state)
        reward_ls.append(total_reward)
        action_ls.append(action)
        is_term_ls.append(done)

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            print(i_episode)
            print_counter = 0
            resulting_time_ls.append(format(t + 1))
            break


    mean = statistics.mean(reward_ls)

    for i in range(len(reward_ls)):
        reward_ls[i] = mean - reward_ls[i]


    pre_state_tensor = torch.reshape(torch.FloatTensor(pre_state_ls),[len(pre_state_ls), 4])
    post_state_tensor = torch.reshape(torch.FloatTensor(post_state_ls), [len(post_state_ls), 4])
    reward_tensor = torch.reshape(torch.FloatTensor(reward_ls), [len(reward_ls), 1])
    action_tensor = torch.reshape(torch.LongTensor(action_ls), [len(action_ls), 1])
    is_term_tensor = torch.reshape(torch.FloatTensor(is_term_ls), [len(is_term_ls), 1])

    #funct.update_net(net,optimizer_net,pre_state_tensor,post_state_tensor,reward_tensor,action_tensor,gamma)
    funct.update_net(policy_net, value_net, policy_opt, value_opt, pre_state_tensor, post_state_tensor, reward_tensor, action_tensor, gamma)


env.close()





# plot
fig, ax = plt.subplots()
x = []
y = []
sum = 0
max_y = 0

moving_avg = []
moving_avg_width = 20

for i in range(len(resulting_time_ls)):
    x.append(i)
    y.append(int(resulting_time_ls[i]))

    sum += int(resulting_time_ls[i])

    if(int(resulting_time_ls[i]) > max_y):
        max_y = int(resulting_time_ls[i])

    if(i > moving_avg_width):
        moving_avg_sum = 0
        for n in range(moving_avg_width):
            moving_avg_sum += int(resulting_time_ls[i - n])
        moving_avg.append(moving_avg_sum/moving_avg_width)
    else:
        moving_avg.append(0)


avg = sum/len(resulting_time_ls)


ax.bar(x, y, width=1, edgecolor="white", linewidth=0.7)
ax.plot(x, moving_avg, linewidth=2.0, color = 'g')
ax.set(xlim=(0, number_of_episode), xticks=np.arange(1, number_of_episode), ylim=(0, (max_y + 5)), yticks=np.arange(1, (max_y + 5)))


#print(resulting_time_ls)
print("avg: ",avg)

plt.show()
