import torch
from torch import nn
import torch.nn.functional as F
import random

import numpy


class policy_net_(torch.nn.Module):
    def __init__(self, n_feature, n_hidden_actor, n_output_actor):
        super(policy_net_, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden_actor)  # hidden layer
        self.hidden2 = nn.Linear(n_hidden_actor, n_hidden_actor)  # hidden layer
        self.predict = nn.Linear(n_hidden_actor, n_output_actor)  # output layer

    def forward(self, state):
        actions = F.leaky_relu(self.hidden1(state.clone()))      # activation function for hidden layer
        actions = F.leaky_relu(self.hidden2(actions))
        actions = self.predict(actions)             # linear output
        softmax = nn.Softmax(dim=-1)
        action_probs = softmax(actions)
        return action_probs

class value_net_(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(value_net_, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden)  # hidden layer
        self.hidden2 = nn.Linear(n_hidden, n_hidden)  # hidden layer
        self.predict = nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, state):
        value = F.relu(self.hidden1(state.clone()))  # activation function for hidden layer
        value = F.relu(self.hidden2(value))
        value = self.predict(value)  # linear output
        return value



def policy(state_, action_ls_, policy_net_):
    with torch.no_grad():
        act_probs = policy_net_(state_)
    return random.choices(action_ls_,act_probs.tolist(),k=1)[0]

def update_net(policy_net_,value_net_, policy_opt_, value_opt_,pre_state_tensor_,post_state_tensor_,reward_tensor_,action_tensor_, gamma_):
    with torch.no_grad():
         current_V = value_net_(post_state_tensor_)
    actor_prob = policy_net_(pre_state_tensor_)
    post_V = value_net_(pre_state_tensor_)
    #print(_, current_V)
    Q = reward_tensor_.clone() + gamma_ * current_V.clone()

    A = Q.clone() - post_V.clone()

    #print(Q,post_V)

    loss_value_fn = torch.nn.MSELoss()
    loss_value = loss_value_fn(post_V, Q)

    #print(loss_value)

    act_probs = actor_prob.gather(1, action_tensor_.view(-1, 1)).view(-1)
    act_probs = torch.reshape(act_probs, [act_probs.shape[0],1])

    #print("ahhhhhhhh",A,act_probs)

    loss_actor = A * torch.log(act_probs)

    #print(loss_actor.shape,loss_value.shape)

    loss_actor = -loss_actor.mean()
    #loss_value = loss_value.mean()

    policy_opt_.zero_grad()
    value_opt_.zero_grad()

    loss_actor.backward(retain_graph=True)
    loss_value.backward()

    policy_opt_.step()
    value_opt_.step()



"""
class NeuralNetwork(torch.nn.Module):
    def __init__(self, n_feature, n_hidden_actor,  n_output_actor, n_hidden_value, n_output_value):
        super(NeuralNetwork, self).__init__()
        self.hidden1 = nn.Linear(n_feature, n_hidden_actor)   # hidden layer
        self.hidden2 = nn.Linear(n_hidden_actor, n_hidden_actor)  # hidden layer
        self.predict = nn.Linear(n_hidden_actor, n_output_actor)   # output layer

        self.hidden1_1 = nn.Linear(n_feature, n_hidden_value)  # hidden layer
        self.hidden2_1 = nn.Linear(n_hidden_value, n_hidden_value)  # hidden layer
        self.hidden3_1 = nn.Linear(n_hidden_value, n_hidden_value)  # hidden layer
        self.hidden4_1 = nn.Linear(n_hidden_value, n_hidden_value)  # hidden layer
        self.predict_1 = nn.Linear(n_hidden_value, n_output_value)  # output

    def forward(self, state):
        actions = F.relu(self.hidden1(state.clone()))      # activation function for hidden layer
        actions = F.relu(self.hidden2(actions))
        actions = self.predict(actions)             # linear output
        softmax = nn.Softmax(dim=-1)
        action_probs = softmax(actions)

        value = F.relu(self.hidden1_1(state.clone()))  # activation function for hidden layer
        value = F.relu(self.hidden2_1(value))
        value = F.relu(self.hidden3_1(value))
        value = F.relu(self.hidden4_1(value))
        value = self.predict_1(value)

        return action_probs, value


def policy(state_, action_ls_, policy_net_):
    with torch.no_grad():
        act_probs = policy_net_(state_)[0]
    return random.choices(action_ls_,act_probs.tolist(),k=1)[0]

def update_net(net_, opt_,pre_state_tensor_,post_state_tensor_,reward_tensor_,action_tensor_, gamma_):
    with torch.no_grad():
         _, current_V = net_(post_state_tensor_)
    actor_prob, post_V = net_(pre_state_tensor_)
    #print(_, current_V)
    Q = reward_tensor_.clone() + gamma_ * current_V.clone()

    A = Q.clone() - post_V.clone()

    #print(Q,post_V)

    loss_value_fn = torch.nn.MSELoss()
    loss_value = loss_value_fn(post_V, Q)

    #print(loss_value)

    act_probs = actor_prob.gather(1, action_tensor_.view(-1, 1)).view(-1)
    act_probs = torch.reshape(act_probs, [act_probs.shape[0],1])

    #print("ahhhhhhhh",A,act_probs)

    loss_actor = A * torch.log(act_probs)

    #print(loss_actor.shape,loss_value.shape)

    loss_actor = -loss_actor.mean()
    #loss_value = loss_value.mean()

    opt_.zero_grad()

    loss_actor.backward(retain_graph=True)
    loss_value.backward()
    opt_.step()
"""
