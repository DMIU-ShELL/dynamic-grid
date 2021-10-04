#-*- coding: utf-8 -*-
import os
import gym
from gym.utils import seeding
from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np
from torchvision import datasets

def _load_mnist():
    # TODO load dataset without using pytorch
    data_path = os.path.dirname(__file__) + '/mnist/'
    _train_data = datasets.MNIST(root=data_path, train=True, download=True, transform=None)
    _test_data = datasets.MNIST(root=data_path, train=False, download=True, transform=None)
    data = {}
    for img, label in _train_data:
        label = int(label)
        if label not in data.keys(): data[label] = []
        data[label].append(np.expand_dims(np.asarray(img, dtype=np.float32), axis=0))
    for img, label in _test_data:
        label = int(label)
        if label not in data.keys(): data[label] = []
        data[label].append(np.expand_dims(np.asarray(img, dtype=np.float32), axis=0))
    return data

class DynamicGrid(gym.Env):
    ACTION_RIGHT = 0
    ACTION_LEFT = 1
    ACTION_UP = 2
    ACTION_DOWN = 3
    ACTION_UP_RIGHT_DIAGONAL = 4
    ACTION_UP_LEFT_DIAGONAL = 5
    ACTION_DOWN_RIGHT_DIAGONAL = 6
    ACTION_DOWN_LEFT_DIAGONAL = 7

    def __init__(self, grid_height=3, grid_width=3, seed=None):
        self.states = _load_mnist()
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=self.states[0][0].shape)
        self.action_space = Discrete(8)

        self.seed(seed)

        # initialise grid
        self.height = grid_height
        self.width = grid_width
        self.grid = np.arange(self.height*self.width)
        self.np_random.shuffle(self.grid)
        self.grid = self.grid.reshape(self.height, self.width)

        # grid specs
        self.coords = [(i, j) for i in range(self.height) for j in range(self.width)]
        self.agent_coord = None
        self.goal_coord = None
        self.permute_mask = None
        self.episode_steps = 0

        # set goal location
        self.reset_task(goal_location=True)

        self.needs_reset = False

        # TODO introduce reward config (dense or sparse(default))

    def step(self, action):
        if self.needs_reset:
            assert False, 'need to reset environment after episode ends'
        curr_coord = list(self.agent_coord)
        if action == DynamicGrid.ACTION_RIGHT:
            curr_coord[1] += 1
        elif action == DynamicGrid.ACTION_LEFT:
            curr_coord[1] -= 1
        elif action == DynamicGrid.ACTION_UP:
            curr_coord[0] -= 1
        elif action == DynamicGrid.ACTION_DOWN:
            curr_coord[0] += 1
        elif action == DynamicGrid.ACTION_UP_RIGHT_DIAGONAL:
            curr_coord[0] -= 1 # up
            curr_coord[1] += 1 # right
        elif action == DynamicGrid.ACTION_UP_LEFT_DIAGONAL:
            curr_coord[0] -= 1 # up
            curr_coord[1] -= 1 # left
        elif action == DynamicGrid.ACTION_DOWN_RIGHT_DIAGONAL:
            curr_coord[0] += 1 # down
            curr_coord[1] += 1 # right
        elif action == DynamicGrid.ACTION_DOWN_LEFT_DIAGONAL:
            curr_coord[0] += 1 # down
            curr_coord[1] -= 1 # left
        else:
            assert False, 'Invalid action'
        # wrap agent position around. (e.g., if the agent takes a right action at the right
        # corner of the grid, the agent position is wrapped around back to the left corner
        # of the grid
        curr_coord[0] %= self.height
        curr_coord[1] %= self.width
        self.agent_coord = tuple(curr_coord)
            
        done = True if self.agent_coord == self.goal_coord else False
        self.needs_reset = True if done else False
        info = {'goal_pos': self.goal_coord, 'agent_pos': self.agent_coord, 
                'permute_input': False if self.permute_mask is None else True, 
                'needs_reset': self.needs_reset}
        self.episode_steps += 1
        reward = 1. / self.episode_steps if done else 0.
        return self._coord_to_state(self.agent_coord), reward, done, info

    def reset_task(self, goal_location=True, transition_dynamics=False, permute_input=False):
        if transition_dynamics:
            # re-order states in the grid
            self.grid = np.arange(self.height*self.width)
            self.np_random.shuffle(self.grid)
            self.grid = self.grid.reshape(self.height, self.width)
        if goal_location:
            idx = self.np_random.randint(0, len(self.coords))
            self.goal_coord = self.coords[idx]
        if permute_input:
            # get observation/image shape
            label = list(self.states.keys())[0]
            c, h, w = self.states[label][0].shape
            # generate permute mask
            mask_ = np.arange(h*w)
            np.random.shuffle(mask_)
            self.permute_mask = mask_
        else:
            self.permute_mask = None
        return self.reset()

    def reset(self):
        self.episode_steps = 0
        while True: # agent should never be instantiated at the current goal position in the grid
            idx = self.np_random.randint(0, len(self.coords))
            if self.goal_coord != self.coords[idx]:
                self.agent_coord = self.coords[idx]
                break
        self.needs_reset = False
        return self._coord_to_state(self.agent_coord)

    def _coord_to_state(self, agent_coord):
        label = self.grid[agent_coord]
        # retrieve one image from the specific label/class
        idx = self.np_random.randint(0, len(self.states[label]))
        obs = self.states[label][idx]
        if self.permute_mask is not None:
            c, h, w = obs.shape
            obs = obs.reshape(c, -1)
            obs = obs[ : , self.permute_mask] # permutation applied to each image channel
            obs = obs.reshape(c, h, w)
        return obs


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render(self, mode='human'):
        pass

if __name__ == '__main__':
    env = gym.make('DynamicGrid-v0')
    print('env created successfully')

