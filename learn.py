# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:55:20 2020

@author: Connor
"""

import threading

import torch
from torch import nn

from utils import compute_policy_gradient_loss, compute_baseline_loss,\
                compute_entropy_loss, reparameterization, get_mali_reward, \
                gen_loss, pixel_loss, compute_latent_loss, \
                compute_discriminator_loss
from core.vtrace import from_logits

# Global Variable
global_goal_gen_outputs = dict()


def learn(
    flags,
    actor_model,
    actor_goal_gen_model,
    actor_gen_model,
    actor_disc_model,
    actor_enc_model,
    model,
    goal_gen_model,
    gen_model,
    disc_model,
    enc_model,
    batch,
    initial_agent_state,
    initial_agent_goal_state,
    initial_agent_gen_state,
    optimizer,
    goal_gen_optimizer,
    gen_optimizer,
    disc_optimizer,
    enc_optimizer,
    scheduler,
    goal_gen_scheduler,
    gen_scheduler,
    disc_scheduler,
    enc_scheduler,
    step,
    lock=threading.Lock(),  # noqa: B008
):
    """Performs a learning (optimization) step."""
    with lock:
        state = batch['frame'][:-1]
        next_state = batch['frame'][1:].long()
        actions = batch["action"][1:]
        done = batch['done'][:-1]
        
        sampled_z = torch.normal(0, 1, [flags.unroll_length, flags.batch_size, flags.latent_dim]).to(device=flags.device)
        enc_output, _ = enc_model(next_state)
        ns_encoded = reparameterization(enc_output['mu'], enc_output['logvar'], sampled_z)
        
        gen_output_enc, _ = gen_model(state, actions, ns_encoded, done, initial_agent_gen_state)
        disc_pred_real, _ = disc_model(state, next_state, actions)
        
        curiosity_gen = (gen_output_enc['prediction'].detach() != next_state
                              ).float().mean(-1).mean(-1).mean(-1)
        curiosity_disc = (1- disc_pred_real['baseline'].detach())
        curiosity_rewards = flags.cr_scaler*curiosity_gen*curiosity_disc

        gen_output, _ = gen_model(state, actions, sampled_z, done, initial_agent_gen_state)

        
        kl_loss = 0.5 * torch.sum(torch.exp(enc_output['logvar']) + enc_output['mu'] ** 2 - enc_output['logvar'] - 1)
    
        
        disc_pred_fake, _ = disc_model(state, gen_output['prediction'].detach(), actions)
        disc_pred_fake_enc, _ = disc_model(state, gen_output_enc['prediction'].detach(), actions)

        
        gen_target_rewards = get_mali_reward(disc_pred_fake['baseline'])
        
        generator_loss = gen_loss(flags, gen_output['obj_policy_logits'], gen_output['col_policy_logits'], 
                            gen_output['att_policy_logits'], gen_output['prediction'], gen_target_rewards) #\

        
        p_loss = pixel_loss(flags, gen_output_enc['obj_policy_logits'], gen_output_enc['col_policy_logits'], 
                            gen_output_enc['att_policy_logits'], torch.flatten(next_state, 2,3))
        e_loss = flags.lambda_pixel * p_loss \
                        + flags.lambda_kl * kl_loss +  flags.lambda_mali * generator_loss
        
        enc_optimizer.zero_grad()
        gen_optimizer.zero_grad()
        
        e_loss.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(enc_model.parameters(), flags.grad_norm_clipping)
        enc_optimizer.step()
        
        
        _enc_output, _ = enc_model(gen_output['prediction'])
        latent_loss = flags.lambda_latent * compute_latent_loss(torch.flatten(_enc_output['mu'], 0, 1), 
                                                      torch.flatten(sampled_z, 0, 1))
        
        latent_loss.backward()
        nn.utils.clip_grad_norm_(gen_model.parameters(), flags.grad_norm_clipping)
        gen_optimizer.step()
        
        enc_scheduler.step()
        gen_scheduler.step()
        
        d_loss = compute_discriminator_loss(torch.flatten(disc_pred_real['baseline'],0,1),
                                            torch.flatten(disc_pred_fake['baseline'],0,1),
                                            torch.flatten(disc_pred_fake_enc['baseline'],0,1))
        
        disc_optimizer.zero_grad()
        d_loss.backward()
        nn.utils.clip_grad_norm_(disc_model.parameters(), flags.grad_norm_clipping)
        disc_optimizer.step()
        disc_scheduler.step()

        actor_gen_model.load_state_dict(gen_model.state_dict())
        actor_disc_model.load_state_dict(disc_model.state_dict())
        actor_enc_model.load_state_dict(enc_model.state_dict())
        
        
        #navigator learning
        learner_outputs, unused_state = model(batch, initial_agent_state, goal=batch['active_goal'],
                                            goal_pos1=batch['active_goal_pos1'],goal_pos2=batch['active_goal_pos2'])
        goal_gen_outputs, unused_state = goal_gen_model(batch, initial_agent_goal_state, 
                                                    goal=batch['goal_creation_goal'],
                                                    goal_pos1=batch['goal_creation_goal_pos1'],
                                                    goal_pos2=batch['goal_creation_goal_pos2'])

        # Take final value function slice for bootstrapping.
        bootstrap_value = learner_outputs["baseline"][-1]
        goal_gen_bootstrap_value = goal_gen_outputs["goal_gen_baseline"][-1]
        
        goals_aux = goal_gen_model.goal_to_policy_index(batch['active_goal'][:-1],
                        batch['active_goal_pos1'][:-1],batch['active_goal_pos2'][:-1]).long()
        

        # Move from obs[t] -> action[t] to action[t] -> obs[t].
        batch = {key: tensor[1:] for key, tensor in batch.items()}
        learner_outputs = {key: tensor[:-1] for key, tensor in learner_outputs.items()}
        goal_gen_outputs = {key: tensor[:-1] for key, tensor in goal_gen_outputs.items()}
        
        
        intrinsic_rewards = flags.intrinsic_reward_coef * batch['goal_reached'].float()
        

        rewards = batch["episode_win"]
        total_rewards = intrinsic_rewards + curiosity_rewards + rewards
        if flags.reward_clipping == "abs_one":
            clipped_rewards = torch.clamp(total_rewards, -1, 1)
        elif flags.reward_clipping == "none":
            clipped_rewards = total_rewards

        discounts = (~(batch["done"] | batch["goal_done"])).float() * flags.discounting

        vtrace_returns = from_logits(
            behavior_policy_logits=batch["policy_logits"],
            target_policy_logits=learner_outputs["policy_logits"],
            actions=batch["action"],
            discounts=discounts,
            rewards=clipped_rewards,
            values=learner_outputs["baseline"],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            learner_outputs["policy_logits"],
            batch["action"],
            vtrace_returns.pg_advantages,
        )
        baseline_loss = flags.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - learner_outputs["baseline"]
        )
        entropy_loss = flags.entropy_cost * compute_entropy_loss(
            learner_outputs["policy_logits"]
        )

        total_loss = pg_loss + baseline_loss + entropy_loss

        episode_returns = batch["episode_return"][batch["done"]]
        if torch.isnan(torch.mean(episode_returns)):
            aux_mean_episode = 0.0
        else:
            aux_mean_episode = torch.mean(episode_returns).item()
            
        stats = {
            "episode_returns": tuple(episode_returns.cpu().numpy()),
            "mean_episode_return": aux_mean_episode, 
            "total_loss": total_loss.item(),
            "pg_loss": pg_loss.item(),
            "baseline_loss": baseline_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "pixel_loss": p_loss.item(),
            "goal_gen_rewards": None,  
            "gg_loss": None,
            "goal_gen_entropy_loss": None,
            "goal_gen_baseline_loss": None,
            "mean_intrinsic_rewards": None,
            "mean_episode_steps": None,
            "ex_reward": None,
        }

        optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), flags.grad_norm_clipping)
        optimizer.step()
        scheduler.step()

        actor_model.load_state_dict(model.state_dict())
        
        #sub goal network learning
        generator_rewards = (rewards 
                             * (flags.npg_discount* (~batch['goal_reached']).float()
                             + batch['goal_reached'].float())
                             * flags.discounting**batch['goals_failed']
                             + curiosity_rewards
                             ).to(device=flags.device)
        

        if flags.reward_clipping == "abs_one":
            generator_clipped_rewards = torch.clamp(generator_rewards, -1, 1)
        else:
            generator_clipped_rewards = generator_rewards



        generator_discounts = (~batch["done"]).float() * flags.discounting
        

        generator_vtrace_returns = from_logits(
            behavior_policy_logits=batch["goal_gen_policy_logits"],
            target_policy_logits=goal_gen_outputs["goal_gen_policy_logits"],
            actions=goals_aux,
            discounts=generator_discounts,
            rewards=generator_clipped_rewards,
            values=goal_gen_outputs["goal_gen_baseline"],
            bootstrap_value=goal_gen_bootstrap_value,
        )   
        


        # Generator Loss
        gg_loss = compute_policy_gradient_loss(
            goal_gen_outputs["goal_gen_policy_logits"],
            goals_aux,
            generator_vtrace_returns.pg_advantages,
        )


        generator_baseline_loss = flags.baseline_cost * compute_baseline_loss(
            generator_vtrace_returns.vs - goal_gen_outputs["goal_gen_baseline"]
        )

        generator_entropy_loss = flags.generator_entropy_cost * compute_entropy_loss(
            goal_gen_outputs["goal_gen_policy_logits"]
        )


        generator_total_loss = generator_entropy_loss + gg_loss + generator_baseline_loss  \


        intrinsic_rewards_gen = batch['goal_done']*(1- 0.9 * (batch["episode_step"].float()/flags.episode_step_limit))
        stats["goal_gen_rewards"] = torch.mean(generator_clipped_rewards).item()  
        stats["gg_loss"] = gg_loss.item() 
        stats["goal_gen_entropy_loss"] = generator_baseline_loss.item() 
        stats["goal_gen_baseline_loss"] = generator_entropy_loss.item() 
        stats["mean_intrinsic_rewards"] = torch.mean(intrinsic_rewards_gen).item()
        stats["mean_episode_steps"] = torch.mean(batch["episode_step"].float()).item()
        stats["ex_reward"] = torch.mean(batch['reward']).item()
        
        
        
        goal_gen_optimizer.zero_grad()
        generator_total_loss.backward()
        
        nn.utils.clip_grad_norm_(goal_gen_model.parameters(), 40.0)
        goal_gen_optimizer.step()
        goal_gen_scheduler.step()
        
        actor_goal_gen_model.load_state_dict(goal_gen_model.state_dict())
        
        

        
        return stats
    
