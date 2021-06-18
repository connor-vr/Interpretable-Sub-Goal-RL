# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:35:17 2020

@author: Connor
"""
import torch
import logging
import os
import traceback

from torch import multiprocessing as mp

from env_utils import create_env, Observation_WrapperSetup
from goal_manager import Goal_Manager
from core.prof import Timings


def act(
    flags,
    actor_index: int,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    model: torch.nn.Module,
    goal_gen_model: torch.nn.Module,
    gen_model: torch.nn.Module,
    enc_model: torch.nn.Module,
    buffers,
    initial_agent_state_buffers,
    initial_agent_goal_state_buffers,
    initial_agent_gen_state_buffers
):
    try:
        logging.info("Actor %i started.", actor_index)
        timings = Timings()  # Keep track of how fast things are.

        env = create_env(flags)
        seed = actor_index ^ int.from_bytes(os.urandom(4), byteorder="little")
        env.seed(seed)
        env.max_steps = flags.episode_step_limit

        env = Observation_WrapperSetup(env)
        env_output = env.initial()
        

        agent_state = model.initial_state(batch_size=1)
        agent_goal_state = goal_gen_model.initial_state(batch_size=1)
        agent_gen_state = gen_model.initial_state(batch_size=1)
        
        goal = torch.LongTensor([[flags.goal[0]]])
        goal_pos1 = torch.LongTensor([[flags.goal[1][0]]])
        goal_pos2 = torch.LongTensor([[flags.goal[1][1]]])
                
        goal_manager = Goal_Manager(goal, goal_pos1, goal_pos2,flags)
        
        goal_creation_goal = goal
        goal_creation_goal_pos1 = goal_pos1
        goal_creation_goal_pos2 = goal_pos2
                
        agent_goal_output, unused_state = goal_gen_model(env_output, agent_goal_state, goal,
                                            goal_pos1, goal_pos2)
        
        
        agent_output, unused_state = model(env_output, agent_state, goal,
                                            goal_pos1, goal_pos2)
        
        
        goal_step_limit_exceeded = False
        goal_done = torch.BoolTensor([[False]])
        goals_failed = 0
        goal_created_timestep = 0
        
        #TODO add support for goal depth > 1
        completed_goals = []
        


        
        while True:
            index = free_queue.get()
            if index is None:
                break

            # Write old rollout end.
            for key in env_output:
                buffers[key][index][0, ...] = env_output[key]
            for key in agent_output:
                buffers[key][index][0, ...] = agent_output[key]
            for key in agent_goal_output:
                buffers[key][index][0, ...] = agent_goal_output[key]

            buffers["goal_done"][index][0, ...] = goal_done
            buffers["goals_failed"][index][0, ...] = goals_failed
            buffers["steps_with_goal"][index][0, ...] = 0
            buffers["goal_reached"][index][0, ...] = torch.BoolTensor([[False]])
            
            for i, tensor in enumerate(agent_state):
                initial_agent_state_buffers[index][i][...] = tensor
            for i, tensor in enumerate(agent_goal_state):
                initial_agent_goal_state_buffers[index][i][...] = tensor
            for i, tensor in enumerate(agent_gen_state):
                initial_agent_gen_state_buffers[index][i][...] = tensor
                
            # Do new rollout.
            for t in range(flags.unroll_length):
                timings.reset()
                
                    
                goal_done = torch.BoolTensor([[False]])
                if env_output['done'][0] == 1:
                    goal, goal_pos1, goal_pos2 = goal_manager.reset()
                    goals_failed = 0
                    completed_goals = []

                    
                with torch.no_grad():
                    agent_goal_output, agent_goal_state = goal_gen_model(env_output, agent_goal_state, 
                                        goal_creation_goal, goal_creation_goal_pos1, goal_creation_goal_pos2)
     
                
                if len(goal_manager.future_goals)<goal_manager.future_goals_limit:
                    goal_created_timestep = t+1
                    
                    goal_manager.add_goal(agent_goal_output["goal"], 
                                     agent_goal_output["goal_pos1"], 
                                     agent_goal_output["goal_pos2"],
                                     )
                    goal = agent_goal_output["goal"]
                    goal_pos1 = agent_goal_output["goal_pos1"]
                    goal_pos2 = agent_goal_output["goal_pos2"]
                
                
                with torch.no_grad():
                    agent_output, agent_state = model(env_output, agent_state, goal,
                                            goal_pos1, goal_pos2)

                timings.time("model")

                env_output = env.step(agent_output["action"])
                
                goal_step_limit_exceeded = goal_manager.step()
                    
                goal_idx, goal_reached = goal_manager.check_goal_reached(env_output['frame'], 
                                                    env_output['done'], env_output['reward'])

                
                if goal_reached or goal_step_limit_exceeded:
                    goal_done = torch.BoolTensor([[True]])
                if not goal_reached and goal_step_limit_exceeded:
                    goals_failed += 1
                

                timings.time("step")

                for key in env_output:
                    buffers[key][index][t + 1, ...] = env_output[key]
                for key in agent_output:
                    buffers[key][index][t + 1, ...] = agent_output[key]
                for key in agent_goal_output:
                    buffers[key][index][t + 1, ...] = agent_goal_output[key]

                buffers["active_goal"][index][t, ...] = goal
                buffers["active_goal_pos1"][index][t, ...] = goal_pos1
                buffers["active_goal_pos2"][index][t, ...] = goal_pos2
                buffers["goal_creation_goal"][index][t, ...] = goal_creation_goal
                buffers["goal_creation_goal_pos1"][index][t, ...] = goal_creation_goal_pos1
                buffers["goal_creation_goal_pos2"][index][t, ...] = goal_creation_goal_pos2
                buffers["goal_done"][index][t + 1, ...] = goal_done
                buffers["goals_failed"][index][t + 1, ...] = goals_failed
                buffers["steps_with_goal"][index][t + 1, ...] = goal_manager.steps_with_goal
                buffers["goal_reached"][index][t + 1, ...] = goal_reached and goal_idx == len(goal_manager.future_goals)
                 
                
                if goal_reached:
                    if goal_idx > 0:
                        completed_goals.append([goal[0][0].item(), [goal_pos1[0][0].item(), goal_pos2[0][0].item()]])
                    goal, goal_pos1, goal_pos2 = goal_manager.goal_reached(goal_idx)
                elif goal_step_limit_exceeded:
                    goal, goal_pos1, goal_pos2 = goal_manager.goal_not_reached()
                    buffers["successful_sub_goal"][index][t + 2 - goal_manager.step_limit:t + 2, ...] = torch.BoolTensor([[False]])
                    
                if env_output['done'][0] == 1 and env_output['reward'] > 0:
                    buffers["override"][index][goal_created_timestep:t + 2, ...] = torch.BoolTensor([[True]])
    

                timings.time("write")
            buffers["active_goal"][index][flags.unroll_length, ...] = goal
            buffers["active_goal_pos1"][index][flags.unroll_length, ...] = goal_pos1
            buffers["active_goal_pos2"][index][flags.unroll_length, ...] = goal_pos2
            buffers["goal_creation_goal"][index][flags.unroll_length, ...] = goal_creation_goal
            buffers["goal_creation_goal_pos1"][index][flags.unroll_length, ...] = goal_creation_goal_pos1
            buffers["goal_creation_goal_pos2"][index][flags.unroll_length, ...] = goal_creation_goal_pos2
            full_queue.put(index)

        if actor_index == 0:
            logging.info("Actor %i: %s", actor_index, timings.summary())

    except KeyboardInterrupt:
        pass  # Return silently.
    except Exception as e:
        logging.error("Exception in worker process %i", actor_index)
        traceback.print_exc()
        print()
        raise e
