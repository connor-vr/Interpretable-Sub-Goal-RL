# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 15:44:26 2020

@author: Connor
"""

import logging
import os
import pprint
import threading
import time
import timeit

os.environ["OMP_NUM_THREADS"] = "1"  # Necessary for multithreading.

import torch
from torch import multiprocessing as mp


from utils import create_buffers, get_batch
from env_utils import create_env, get_env_dim
from actor import act
from networks import Net
from GAN_nets import Generator, Discriminator, Encoder
from learn import learn
from core.file_writer import FileWriter
from core.prof import Timings




def train(flags):  # pylint: disable=too-many-branches, too-many-statements
    if flags.xpid is None:
        flags.xpid = "torchbeast-%s" % time.strftime("%Y%m%d-%H%M%S")
        if flags.load:
            load_checkpointpath = "./latest/model.tar"
    else:
        if flags.load:
            load_checkpointpath = os.path.expandvars(
                os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar")))
    plogger = FileWriter(
        xpid=flags.xpid, xp_args=flags.__dict__, rootdir=flags.savedir
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser("%s/%s/%s" % (flags.savedir, flags.xpid, "model.tar"))
    )

    if flags.num_buffers is None:  # Set sensible default for num_buffers.
        flags.num_buffers = max(2 * flags.num_actors, flags.batch_size)
    if flags.num_actors >= flags.num_buffers:
        raise ValueError("num_buffers should be larger than num_actors")
    if flags.num_buffers < flags.batch_size:
        raise ValueError("num_buffers should be larger than batch_size")

    T = flags.unroll_length
    B = flags.batch_size

    flags.device = None
    if not flags.disable_cuda and torch.cuda.is_available():
        logging.info("Using CUDA.")
        flags.device = torch.device("cuda")
    else:
        logging.info("Not using CUDA.")
        flags.device = torch.device("cpu")

    env = create_env(flags)
    
    
    state_dim, action_dim = get_env_dim(env)
    model = Net(state_dim, action_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision, flags.use_lstm)
    sub_goal_dim = state_dim*flags.num_objects
    goal_gen_model = Net(state_dim, sub_goal_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision, flags.use_lstm, goal_gen=True)
    gen_model = Generator(state_dim, action_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision, flags.latent_dim, use_lstm=False)
    disc_model = Discriminator(state_dim, action_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision)
    enc_model = Encoder(state_dim, flags.hidden_dim, flags.latent_dim,
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision)
    
    if flags.load:
        load_checkpoint = torch.load(load_checkpointpath, map_location="cpu")
        model.load_state_dict(load_checkpoint["model_state_dict"])
        goal_gen_model.load_state_dict(load_checkpoint["goal_gen_model_state_dict"])
        
        gen_model.load_state_dict(load_checkpoint["gen_model_state_dict"])
        disc_model.load_state_dict(load_checkpoint["disc_model_state_dict"])
        enc_model.load_state_dict(load_checkpoint["enc_model_state_dict"])
    
    buffers = create_buffers(flags, env.observation_space.shape, model.output_dim, goal_gen_model.output_dim)

    model.share_memory()
    goal_gen_model.share_memory()
    gen_model.share_memory()
    disc_model.share_memory()
    enc_model.share_memory()

    # Add initial RNN state.
    initial_agent_state_buffers = []
    initial_agent_goal_state_buffers = []
    initial_agent_gen_state_buffers = []
    for _ in range(flags.num_buffers):
        state = model.initial_state(batch_size=1)
        state_goal = goal_gen_model.initial_state(batch_size=1)
        state_gen = gen_model.initial_state(batch_size=1)
        for t, t2, t3 in zip(state, state_goal, state_gen):
            t.share_memory_()
            t2.share_memory_()
            t3.share_memory_()
        initial_agent_state_buffers.append(state)
        initial_agent_goal_state_buffers.append(state_goal)
        initial_agent_gen_state_buffers.append(state_gen)

    actor_processes = []
    ctx = mp.get_context()
    free_queue = ctx.SimpleQueue()
    full_queue = ctx.SimpleQueue()

    for i in range(flags.num_actors):
        actor = ctx.Process(
            target=act,
            args=(
                flags,
                i,
                free_queue,
                full_queue,
                model,
                goal_gen_model,
                gen_model,
                enc_model,
                buffers,
                initial_agent_state_buffers,
                initial_agent_goal_state_buffers,
                initial_agent_gen_state_buffers
            ),
        )
        actor.start()
        actor_processes.append(actor)

    learner_model = Net(state_dim, action_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision, flags.use_lstm).to(device=flags.device)
    learner_goal_gen_model = Net(state_dim, sub_goal_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision, flags.use_lstm, goal_gen=True).to(device=flags.device)
    learner_gen_model = Generator(state_dim, action_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision, flags.latent_dim, use_lstm=False).to(device=flags.device)
    learner_disc_model = Discriminator(state_dim, action_dim, flags.hidden_dim, 
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision).to(device=flags.device)
    learner_enc_model = Encoder(state_dim, flags.hidden_dim, flags.latent_dim,
                flags.num_objects,  flags.embedding_dim, flags.num_colours, flags.num_attributes, 
                flags.secondary_embedding_dim, flags.vision).to(device=flags.device)

    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )
    
    goal_gen_optimizer = torch.optim.RMSprop(
        learner_goal_gen_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )
    
    gen_optimizer = torch.optim.RMSprop(
        learner_gen_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )
    
    disc_optimizer = torch.optim.RMSprop(
        learner_disc_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )
    
    enc_optimizer = torch.optim.RMSprop(
        learner_enc_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, flags.total_steps) / flags.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    goal_gen_scheduler = torch.optim.lr_scheduler.LambdaLR(goal_gen_optimizer, lr_lambda)
    gen_scheduler = torch.optim.lr_scheduler.LambdaLR(gen_optimizer, lr_lambda)
    disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_lambda)
    enc_scheduler = torch.optim.lr_scheduler.LambdaLR(enc_optimizer, lr_lambda)
    
    if flags.load:
        learner_model.load_state_dict(load_checkpoint["model_state_dict"])
        learner_goal_gen_model.load_state_dict(load_checkpoint["goal_gen_model_state_dict"])
        
        learner_gen_model.load_state_dict(load_checkpoint["gen_model_state_dict"])
        learner_disc_model.load_state_dict(load_checkpoint["disc_model_state_dict"])
        learner_enc_model.load_state_dict(load_checkpoint["enc_model_state_dict"])
        

    logger = logging.getLogger("logfile")
    stat_keys = [
        "total_loss",
        "mean_episode_return",
        "pg_loss",
        "baseline_loss",
        "entropy_loss",
        "goal_gen_rewards",  
        "gg_loss",
        "goal_gen_entropy_loss",
        "goal_gen_baseline_loss",
        "pixel_loss",
        "mean_intrinsic_rewards",
        "mean_episode_steps",
        "ex_reward",
    ]
    logger.info("# Step\t%s", "\t".join(stat_keys))

    step, stats = 0, {}

    def batch_and_learn(i, lock=threading.Lock()):
        """Thread target for the learning process."""
        nonlocal step, stats
        timings = Timings()
        while step < flags.total_steps:
            timings.reset()
            batch, agent_state, agent_goal_state, agent_gen_state = get_batch(
                flags,
                free_queue,
                full_queue,
                buffers,
                initial_agent_state_buffers,
                initial_agent_goal_state_buffers,
                initial_agent_gen_state_buffers,
                timings,
            )
            stats = learn(
                flags, model, goal_gen_model, gen_model, disc_model, enc_model, 
                learner_model, learner_goal_gen_model, learner_gen_model, learner_disc_model, learner_enc_model, 
                batch, agent_state, agent_goal_state, agent_gen_state, 
                optimizer, goal_gen_optimizer, gen_optimizer, disc_optimizer, enc_optimizer, 
                scheduler, goal_gen_scheduler, gen_scheduler, disc_scheduler, enc_scheduler,
                step
            )
            timings.time("learn")
            with lock:
                to_log = dict(step=step)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                step += T * B

        if i == 0:
            logging.info("Batch and learn: %s", timings.summary())

    for m in range(flags.num_buffers):
        free_queue.put(m)

    threads = []
    for i in range(flags.num_learner_threads):
        thread = threading.Thread(
            target=batch_and_learn, name="batch-and-learn-%d" % i, args=(i,)
        )
        thread.start()
        threads.append(thread)

    def checkpoint():
        if flags.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "goal_gen_model_state_dict": goal_gen_model.state_dict(),
                "gen_model_state_dict": gen_model.state_dict(),
                "disc_model_state_dict": disc_model.state_dict(),
                "enc_model_state_dict": enc_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "goal_gen_optimizer_state_dict": goal_gen_optimizer.state_dict(),
                "gen_optimizer_state_dict": gen_optimizer.state_dict(),
                "disc_optimizer_state_dict": disc_optimizer.state_dict(),
                "enc_optimizer_state_dict": enc_optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "goal_gen_scheduler_state_dict": goal_gen_scheduler.state_dict(),
                "gen_scheduler_state_dict": gen_scheduler.state_dict(),
                "disc_scheduler_state_dict": disc_scheduler.state_dict(),
                "enc_scheduler_state_dict": enc_scheduler.state_dict(),
                "flags": vars(flags),
            },
            checkpointpath,
        )
        

    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()
        while step < flags.total_steps:
            start_step = step
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > 10 * 60:  # Save every 10 min.
                checkpoint()
                last_checkpoint_time = timer()

            sps = (step - start_step) / (timer() - start_time)
            if stats.get("episode_returns", None):
                mean_return = (
                    "Return per episode: %.1f. " % stats["mean_episode_return"]
                )
            else:
                mean_return = ""
            total_loss = stats.get("total_loss", float("inf"))
            logging.info(
                "Steps %i @ %.1f SPS. Loss %f. %sStats:\n%s",
                step,
                sps,
                total_loss,
                mean_return,
                pprint.pformat(stats),
            )
    except KeyboardInterrupt:
        return  # Try joining actors then quit.
    else:
        for thread in threads:
            thread.join()
        logging.info("Learning finished after %d steps.", step)
    finally:
        for _ in range(flags.num_actors):
            free_queue.put(None)
        for actor in actor_processes:
            actor.join(timeout=1)

    checkpoint()
    plogger.close()