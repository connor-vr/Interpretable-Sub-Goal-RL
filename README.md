# A Novel Approach to Curiosity and Explainable Reinforcement Learning via Interpretable Sub-Goals

This repository contain the code used by the authors to produce the results published with [A Novel Approach to Curiosity and Explainable Reinforcement Learning via Interpretable Sub-Goals](https://arxiv.org/abs/2104.0663)

## Set up
Package versions as used by the authors:
- gym==0.15.7
- gym-minigrid==1.0.1
- numpy==1.19.2
- torch==1.6.0

Once minigrid installed, replace gym_minigrid/envs/doorkey.py with the version contained within this repository to use the custom enviroments created by the authors. 

## Running the code
Defaults are set within the main.py file. Simply run:
```
python main.py
```
to use the default experiment settings. Alternatively, here is an example using non default options:
```
python main.py --env MiniGrid-DoorKey-5x5-v0 --num_actors 8 --total_steps 5000000 
```
