from gym.envs.registration import register

register(
    id='DynamicGrid-v0',
    entry_point='dynamic_grid.env:DynamicGrid',
    max_episode_steps=10
)

from .env import DynamicGrid
