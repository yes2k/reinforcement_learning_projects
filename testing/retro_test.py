import retro
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from random import randint
from torch.autograd import Variable
import random
import matplotlib.pyplot as plt
import cv2
from gym.envs.classic_control import rendering

class DQL():
    def __init__(self, lr):
        # input size is batch_size, 4, 100, 100
        self.model = torch.nn.Sequential(
                        torch.nn.Conv2d(4, 32, 5),
                        torch.nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        torch.nn.Conv2d(32, 64, 5),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(kernel_size=2),
                        nn.Flatten(),
                        nn.Linear(64*22*22, 10000),
                        torch.nn.ReLU(),
                        nn.Linear(10000, 100),
                        torch.nn.ReLU(),
                        nn.Linear(100, 12)
                )
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr)

    def update(self, state, y):
        print("updating")
        y_pred = self.model(state)
        print("got prediction")
        loss = self.criterion(y_pred, Variable(torch.Tensor(y)))
        print("got loss")
        self.optimizer.zero_grad()
        loss.backward()
        print("got backward pass")
        self.optimizer.step()
        print("finished updating")

    def predict(self, state):
        # print(state.size())
        with torch.no_grad():
            return self.model(state)

    def replay(self, memory, size, gamma=0.9):
        # print("on replay")
        if len(memory) >= size:
            states = []
            targets = []

            batch = random.sample(memory, size)

            for state, action, next_state, reward, done in batch:
                states.append(state)

                q_values = self.predict(state).tolist()[0]
                if done:
                    q_values[action] = reward
                else:
                    q_values_next = self.predict(next_state)
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                targets.append(q_values)
            self.update(torch.cat(states, dim=0), targets)


def q_learning(env, model, episodes, gamma=0.9,
               epsilon=0.3, eps_decay=0.99,
               replay=True, replay_size=20):
    final = []
    memory = []
    curr_n_states = []
    next_n_states = []
    for episode in range(episodes):
        print(episode)

        state = env.reset()
        state = cv2.resize(cv2.cvtColor(state, cv2.COLOR_RGB2GRAY), (100, 100))
        curr_n_states.append(torch.Tensor(state))
        done = False
        total = 0
        
        while not done:
            env.render()
            if len(curr_n_states) < 4:
                state, _, _, _ = env.step(env.action_space.sample())
                curr_n_states.append(torch.Tensor(cv2.resize(cv2.cvtColor(
                                    state, cv2.COLOR_RGB2GRAY), (100, 100))))
            else:
                if random.random() < epsilon:
                    # action = env.action_space.sample()
                    action = random.randint(0, 12)
                    env_action = [1 if i == action else 0 for i in range(0, 12)]
                else:
                    q_values = model.predict(torch.stack(curr_n_states, dim=0).unsqueeze(0))
                    action = torch.argmax(q_values).item()
                    env_action = [1 if i == action else 0 for i in range(0, 12)]

                next_state, reward, done, _ = env.step(env_action)

                next_n_states = [curr_n_states[1], curr_n_states[2],
                        curr_n_states[3],torch.Tensor(cv2.resize(
                        cv2.cvtColor(next_state, cv2.COLOR_RGB2GRAY),
                                                    (100,100)))]

                # print(torch.stack(next_n_states, dim=0).unsqueeze(0).size())

                total += reward
                memory.append((torch.stack(curr_n_states, dim=0).unsqueeze(0), action,
                        torch.stack(next_n_states, dim=0).unsqueeze(0), reward, done))
                q_values = model.predict(torch.stack(curr_n_states,
                                                        dim=0).unsqueeze(0)).tolist()

                if done:
                    if not replay:
                        q_values[action] = reward
                        model.update(torch.stack(curr_n_states, dim=0).unsqueeze(0),
                                                                    q_values)

                if replay:
                    model.replay(memory, replay_size, gamma)
                else:
                    q_values_next = model.predict(torch.stack(next_n_states, dim=0).unsqueeze(0))
                    q_values[action] = reward + gamma * torch.max(q_values_next).item()
                    model.update(torch.stack(curr_n_states, dim=0).unsqueeze(0), q_values)

                curr_n_states = next_n_states

            # Update epsilon
            epsilon = max(epsilon * eps_decay, 0.01)
            final.append(total)
    return final


env = retro.make(game='Airstriker-Genesis')
print(env.action_space.sample())
ql_model = DQL(0.001)
q_learning(env, ql_model, 10)

# def main():
#     # viewer = rendering.SimpleImageViewer()
#     env = retro.make(game='Airstriker-Genesis')
#     obs = env.reset()
#     while True:
#         step = np.zeros(12)
#         step[randint(0,11)] = 1
#         obs, rew, done, info = env.step(step)
#         # downscaled = cv2.resize(obs, (100, 100))
#         # bnw = cv2.cvtColor(downscaled, cv2.COLOR_RGB2GRAY)
#         # viewer.imshow(downscaled)
#         if done:
#             # obs = env.reset()
#             env.close()

# if __name__ == "__main__":
#     plt.figure()
#     main()
