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
        # input size is 1, 100, 100
        self.model = torch.nn.Sequential(
                        torch.nn.Conv2d(4, 32, 5),
                        torch.nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2),
                        torch.nn.Conv2d(32, 64, 5),
                        torch.nn.ReLU(),
                        torch.nn.MaxPool2d(kernel_size=2), # outputs a shape of 1, 64, 22, 22
                        nn.Linear(64*22*22, 10000),
                        torch.nn.Relu(),
                        nn.Linear(10000, 100),
                        torch.nn.ReLU(),
                        nn.Linear(100, 12)
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



# env = retro.make(game="Airstriker-Genesis")
#
# n_state = env.observation_space.shape
# # Number of actions
# n_action = env.action_space
#
# print(n_state)
# print(n_action)

def main():
    viewer = rendering.SimpleImageViewer()
    env = retro.make(game='Airstriker-Genesis')
    obs = env.reset()
    while True:
        step = np.zeros(12)
        step[randint(0,11)] = 1
        obs, rew, done, info = env.step(step)
        downscaled = cv2.resize(obs, (100, 100))
        viewer.imshow(downscaled)
        if done:
            # obs = env.reset()
            env.close()

if __name__ == "__main__":
    plt.figure()
    main()
