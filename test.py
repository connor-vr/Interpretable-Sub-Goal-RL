# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:55:19 2020

@author: Connor
"""

import logging
import os
import torch

from goal_manager import Goal_Manager
from env_utils import create_env, get_env_dim, Observation_WrapperSetup
from utils import reparameterization

from networks import Net
from GAN_nets import Generator, Discriminator, Encoder

def test(flags, num_episodes: int = 1):
    if flags.xpid is None:
        checkpointpath = "./latest/model.tar"
    else:
        checkpointpath = os.path.expandvars(
            os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
        )

    env = create_env(flags)
    env.max_steps = flags.episode_step_limit
    env.reset()
    state_dim, action_dim = get_env_dim(env)
    seed = int.from_bytes(os.urandom(4), byteorder="little")
    env.seed(seed)
    env = Observation_WrapperSetup(env)
    model = Net(state_dim, action_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision, flags.use_lstm)
    model.eval()
    sub_goal_dim = state_dim*flags.num_objects
    goal_gen_model = Net(state_dim, sub_goal_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision, flags.use_lstm, goal_gen=True)
    goal_gen_model.eval()
    gen_model = Generator(state_dim, action_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision, flags.latent_dim, use_lstm=False)
    gen_model.eval()
    disc_model = Discriminator(state_dim, action_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision)
    disc_model.eval()
    enc_model = Encoder(state_dim, flags.hidden_dim, flags.latent_dim,
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision)
    enc_model.eval()
    checkpoint = torch.load(checkpointpath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    goal_gen_model.load_state_dict(checkpoint["goal_gen_model_state_dict"])
    
    gen_model.load_state_dict(checkpoint["gen_model_state_dict"])
    disc_model.load_state_dict(checkpoint["disc_model_state_dict"])
    enc_model.load_state_dict(checkpoint["enc_model_state_dict"])

    observation = env.initial()
    returns = []
    
    goal = torch.LongTensor([[flags.goal[0]]])
    goal_pos1 = torch.LongTensor([[flags.goal[1][0]]])
    goal_pos2 = torch.LongTensor([[flags.goal[1][1]]])
    
    goal_manager = Goal_Manager(goal, goal_pos1, goal_pos2,flags)
    
    agent_state = model.initial_state(batch_size=1)
    agent_goal_state = goal_gen_model.initial_state(batch_size=1)
    agent_gen_state = gen_model.initial_state(batch_size=1)
    agent_gen_state_enc = gen_model.initial_state(batch_size=1)
    
    step = 0
    while len(returns) < num_episodes:
        if flags.mode == "test_render":
            env.gym_env.render()

        step += 1
        agent_goal_output, agent_goal_state = goal_gen_model(observation, agent_goal_state, goal=goal,
                              goal_pos1=goal_pos1,goal_pos2=goal_pos2)
        
        if len(goal_manager.future_goals)<goal_manager.future_goals_limit:
            #create new sub goal
            goal_manager.add_goal(agent_goal_output["goal"], 
                             agent_goal_output["goal_pos1"], 
                             agent_goal_output["goal_pos2"]
                             )
            goal = agent_goal_output["goal"]
            goal_pos1 = agent_goal_output["goal_pos1"]
            goal_pos2 = agent_goal_output["goal_pos2"]
            


        policy_outputs, agent_state = model(observation, agent_state, goal=goal,
                              goal_pos1=goal_pos1,goal_pos2=goal_pos2)
        
        sampled_z = torch.normal(0, 1, [1,1, flags.latent_dim])
        gen_output, agent_gen_state = gen_model(observation['frame'], policy_outputs["action"], sampled_z,
                                                observation['done'], agent_gen_state)

        
        state = observation['frame']

        observation = env.step(policy_outputs["action"])
        goal_step_limit_exceeded = goal_manager.step()
        
        
        enc_output, _ = enc_model(observation['frame'])
        ns_encoded = reparameterization(enc_output['mu'], enc_output['logvar'], sampled_z)
        gen_output_enc, agent_gen_state_enc = gen_model(state, policy_outputs["action"], ns_encoded,
                                                observation['done'], agent_gen_state_enc)
        
        
        disc_pred_real, _ = disc_model(state, observation['frame'], policy_outputs["action"])
        disc_pred_fake, _ = disc_model(state, gen_output['prediction'].detach(), policy_outputs["action"])

        
        goal_idx, goal_reached =  goal_manager.check_goal_reached(observation['frame'], 
                                                    observation['done'], observation['reward'])
        if goal_reached:
            goal, goal_pos1, goal_pos2 = goal_manager.goal_reached(goal_idx)

        elif goal_step_limit_exceeded:
            goal, goal_pos1, goal_pos2 = goal_manager.goal_not_reached()

        if observation["done"].item():
            goal, goal_pos1, goal_pos2 = goal_manager.reset()
            returns.append(observation["episode_return"].item())
            logging.info(
                "Episode ended after %d steps. Return: %.1f",
                observation["episode_step"].item(),
                observation["episode_return"].item(),
            )
    env.close()
    logging.info(
        "Average returns over %i steps: %.1f", num_episodes, sum(returns) / len(returns)
    )