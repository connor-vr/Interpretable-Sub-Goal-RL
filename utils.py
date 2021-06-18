# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:34:15 2020

@author: Connor
"""



import torch
import typing
import threading


from torch.nn import functional as F
from torch import multiprocessing as mp

Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def reparameterization(mu, logvar, sampled_z):
     std = torch.exp(logvar / 2)
     z = sampled_z * std + mu
     return z


def get_mali_reward(disc_out):
     disc_out = disc_out*.999
     rewards = torch.div(disc_out, 1 - disc_out)
     rewards = torch.div(rewards, torch.sum(rewards)+0.01)
     rewards = torch.clamp(rewards, -1, 1)
     rewards -= torch.mean(rewards)
     return rewards.detach()


def gen_loss(flags, logits_obj, logits_col, logits_att, selected, reward):
     selected = torch.flatten(selected, 2,3)
     target_obj_onehot = F.one_hot(selected[:,:,:,0].long(), flags.num_objects).float()
     target_col_onehot = F.one_hot(selected[:,:,:,1].long(), flags.num_colours).float()
     target_att_onehot = F.one_hot(selected[:,:,:,2].long(), flags.num_attributes).float()
     
     pred = (torch.sum(logits_obj * target_obj_onehot, dim=-1) + torch.sum(logits_col * target_col_onehot, dim=-1)
             + torch.sum(logits_att * target_att_onehot, dim=-1)).mean(-1)
     
     loss = -torch.sum(pred * reward)
     return loss


def pixel_loss(flags, logits_obj, logits_col, logits_att, next_state):
     obj_weights = torch.ones(flags.num_objects).cuda()
     obj_weights[flags.goal[0]] = 2.
     logits_obj = torch.flatten(logits_obj, 0,1).permute(0,2,1)
     logits_col = torch.flatten(logits_col, 0,1).permute(0,2,1)
     logits_att = torch.flatten(logits_att, 0,1).permute(0,2,1)
     next_state = torch.flatten(next_state, 0,1)     
     
     loss = F.cross_entropy(logits_obj, next_state[:,:,0], weight=obj_weights) \
            + F.cross_entropy(logits_col, next_state[:,:,1]) \
            + F.cross_entropy(logits_att, next_state[:,:,2])
     return loss

def compute_latent_loss(mu, sampled_z):
    return F.l1_loss(mu, sampled_z)

def compute_discriminator_loss(out_real, out_fake, out_fake_enc):
    real = torch.ones_like(out_real)
    fake = torch.zeros_like(out_fake)
    return 2 * F.mse_loss(out_real, real) \
        + F.mse_loss(out_fake, fake) \
        + F.mse_loss(out_fake_enc, fake)


def get_batch(
    flags,
    free_queue: mp.SimpleQueue,
    full_queue: mp.SimpleQueue,
    buffers: Buffers,
    initial_agent_state_buffers,
    initial_agent_goal_state_buffers,
    initial_agent_gen_state_buffers,
    timings,
    lock=threading.Lock(),
):
    with lock:
        timings.time("lock")
        indices = [full_queue.get() for _ in range(flags.batch_size)]
        timings.time("dequeue")
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1) for key in buffers
    }
    initial_agent_state = (
        torch.cat(ts, dim=1)
        for ts in zip(*[initial_agent_state_buffers[m] for m in indices])
    )
    initial_agent_goal_state = (
        torch.cat(tss, dim=1)
        for tss in zip(*[initial_agent_goal_state_buffers[m] for m in indices])
    )
    initial_agent_gen_state = (
        torch.cat(tss, dim=1)
        for tss in zip(*[initial_agent_gen_state_buffers[m] for m in indices])
    )
    timings.time("batch")
    for m in indices:
        free_queue.put(m)
    timings.time("enqueue")
    batch = {k: t.to(device=flags.device, non_blocking=True) for k, t in batch.items()}
    initial_agent_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_state
    )
    initial_agent_goal_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_goal_state
    )
    initial_agent_gen_state = tuple(
        t.to(device=flags.device, non_blocking=True) for t in initial_agent_gen_state
    )
    timings.time("device")
    return batch, initial_agent_state, initial_agent_goal_state, initial_agent_gen_state


def create_buffers(flags, obs_shape, num_actions, logits_size) -> Buffers:
    T = flags.unroll_length
    specs = dict(
        frame=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        reward=dict(size=(T + 1,), dtype=torch.float32),
        done=dict(size=(T + 1,), dtype=torch.bool),
        episode_return=dict(size=(T + 1,), dtype=torch.float32),
        episode_step=dict(size=(T + 1,), dtype=torch.int32),
        active_goal=dict(size=(T + 1,), dtype=torch.uint8),
        active_goal_pos1=dict(size=(T + 1,), dtype=torch.uint8),
        active_goal_pos2=dict(size=(T + 1,), dtype=torch.uint8),
        policy_logits=dict(size=(T + 1, num_actions), dtype=torch.float32),
        baseline=dict(size=(T + 1,), dtype=torch.float32),
        goal_gen_baseline=dict(size=(T + 1,), dtype=torch.float32),
        action=dict(size=(T + 1,), dtype=torch.int64),
        episode_win=dict(size=(T + 1,), dtype=torch.int32),
        goal_gen_policy_logits=dict(size=(T + 1, logits_size), dtype=torch.float32),
        goal_gen_out=dict(size=(T + 1,), dtype=torch.int64),
        goal=dict(size=(T + 1,), dtype=torch.int64),
        goal_pos1=dict(size=(T + 1,), dtype=torch.int64),
        goal_pos2=dict(size=(T + 1,), dtype=torch.int64),
        goal_creation_goal=dict(size=(T + 1,), dtype=torch.uint8),
        goal_creation_goal_pos1=dict(size=(T + 1,), dtype=torch.uint8),
        goal_creation_goal_pos2=dict(size=(T + 1,), dtype=torch.uint8),
        goal_done=dict(size=(T + 1,), dtype=torch.bool),
        goals_failed=dict(size=(T + 1,), dtype=torch.uint8),
        prediction=dict(size=(T + 1, *obs_shape), dtype=torch.uint8),
        obj_policy_logits=dict(size=(T + 1, flags.vision**2, flags.num_objects), dtype=torch.float32),
        col_policy_logits=dict(size=(T + 1, flags.vision**2, flags.num_colours), dtype=torch.float32),
        att_policy_logits=dict(size=(T + 1, flags.vision**2, flags.num_attributes), dtype=torch.float32),
        successful_sub_goal=dict(size=(T + 1,), dtype=torch.bool),
        steps_with_goal=dict(size=(T + 1,), dtype=torch.uint8),
        override=dict(size=(T + 1,), dtype=torch.bool),
        goal_reached=dict(size=(T + 1,), dtype=torch.bool),
    )
    buffers: Buffers = {key: [] for key in specs}
    for _ in range(flags.num_buffers):
        for key in buffers:
            buffers[key].append(torch.empty(**specs[key]).share_memory_())
    return buffers


