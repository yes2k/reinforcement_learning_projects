import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
from itertools import chain

class DQL():
    def __init__(self, state_dim, hidden_dim, action_dim, lr):
        self.model = torch.nn.Sequential(
                        torch.nn.Linear(state_dim, hidden_dim),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim, hidden_dim*2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(hidden_dim*2, action_dim)
                )
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr)

    def update(self, state, y):
        y_pred = self.model(torch.Tensor(state))
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, state):
        with torch.no_grad():
            return self.model(torch.Tensor(state))


def q_learning(env, model, episodes, num_past_states, gamma=0.9,
               epsilon=0.3, eps_decay=0.99):
    final = []
    for episode in range(episodes):
        print(episode)

        last_n_states = []

        state = env.reset()
        last_n_states.append(state)

        # if there aren't enough past states, we pad them with zeros
        while len(last_n_states) < num_past_states:
            last_n_states.append([0 for i in range(len(last_n_states[0]))])

        done = False
        total = 0
        while not done:
            curr_last_n_states = list(chain.from_iterable(last_n_states))
            print(last_n_states)
            # Choosing random action with probability epsilon (epsilon greedy)
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                q_values = model.predict(curr_last_n_states)
                action = torch.argmax(q_values).item()

            next_state, reward, done, _ = env.step(action)
            env.render()

            total += reward

            q_values = model.predict(curr_last_n_states).tolist()

            if done:
                q_values[action] = reward
                model.update(curr_last_n_states, q_values)
            else:
                last_n_states.pop()
                last_n_states.insert(0, next_state)
                q_values_next = model.predict(list(chain.from_iterable(last_n_states)))
                q_values[action] = reward + gamma * torch.max(q_values_next).item()
                model.update(curr_last_n_states, q_values)
            # state = next_state
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
    return final


env = gym.envs.make("CartPole-v1")

# Number of past states to input into our NN
n_past_states = 1

# Number of states
n_state = env.observation_space.shape[0]*n_past_states

# Number of actions
n_action = env.action_space.n

# Number of episodes
episodes = 150

# Number of hidden nodes in the DQN
n_hidden = 50
# Learning rate
lr = 0.001

simple_dqn = DQL(n_state, n_hidden, n_action, lr)
simple = q_learning(env, simple_dqn, episodes, n_past_states,
                gamma=.9, epsilon=0.3)

plt.figure()
plt.plot(range(episodes), simple)
plt.show()
