# Dynamic Grid World Environment
This is an implementation of a configurable square grid world environment (tasks can be defined based on changes in the reward structure, input space or transition dynamics).

For a 3x3 grid world, the states or input space are defined as MNIST images. 

### Installation / Usage instructions
- Clone repository using `git clone <repo-url>.git`
- Install environment `pip install -e .`
- To use environment run:
```
>>> import dynamic_grid
>>> env = gym.make('DynamicGrid-v0')
```

or run

```
>>> import dynamic_grid
>>> env = dynamic_grid.DynamicGrid()
```
