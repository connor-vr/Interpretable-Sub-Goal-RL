# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging

from train import train
from test import test


# yapf: disable
parser = argparse.ArgumentParser(description="PyTorch Scalable Agent")

parser.add_argument("--env", type=str, 
                    # default='MiniGrid-DoorKey-RandGoal-5x6-v0',
                    # default='MiniGrid-DoorKey-Rand-6x6-v0',
                    # default='MiniGrid-DoorKey-Rand-8x8-v0',
                    # default='MiniGrid-DoorKey-Rand-10x10-v0',
                    # default="MiniGrid-Empty-5x5-v0",
                    default="MiniGrid-DoorKey-5x5-v0",
                    # default="MiniGrid-DoorKey-6x6-v0",
                    help="Gym environment.")
parser.add_argument("--mode", 
                    default="train",
                    # default="test",
                    choices=["train", "test", "test_render"],
                    help="Training or test mode.")
parser.add_argument("--xpid", default=None,
                    help="Experiment id (default: None).")
parser.add_argument("--load", 
                    # action='store_false',
                    action='store_true',
                    help="load previously trained model")

# Training settings.
parser.add_argument("--disable_checkpoint", action="store_true",
                    help="Disable saving checkpoint.")
parser.add_argument("--savedir", default="/logs/",
                    help="Root dir where experiment data will be saved.")
parser.add_argument("--num_actors", default=4, type=int, metavar="N",
                    help="Number of actors (default: 4).")
parser.add_argument("--total_steps", default=30000000, type=int, metavar="T",
                    help="Total environment steps to train for.")
parser.add_argument("--batch_size", default=8, type=int, metavar="B",
                    help="Learner batch size.")
parser.add_argument("--unroll_length", default=100, type=int, metavar="T",
                    help="The unroll length (time dimension).")
parser.add_argument("--num_buffers", default=None, type=int,
                    metavar="N", help="Number of shared-memory buffers.")
parser.add_argument("--num_learner_threads", "--num_threads", default=2, type=int,
                    metavar="N", help="Number learner threads.")
parser.add_argument("--disable_cuda", 
                    action="store_true",
                    # action="store_false",
                    help="Disable CUDA.")
parser.add_argument("--use_lstm", action="store_false",
                    help="Use LSTM in agent model.")
parser.add_argument("--episode_step_limit", default=100, type=int, metavar="L",
                    help="Total steps before episode reset")

# Loss settings.
parser.add_argument("--entropy_cost", default=0.01,
                    type=float, help="Entropy cost/multiplier.")
parser.add_argument("--baseline_cost", default=0.5,
                    type=float, help="Baseline cost/multiplier.")
parser.add_argument('--generator_entropy_cost', default=0.075, type=float,
                    help='Entropy cost/multiplier.')
parser.add_argument("--discounting", default=0.99,
                    type=float, help="Discounting factor.")
parser.add_argument("--reward_clipping", default="abs_one",
                    choices=["abs_one", "none"],
                    help="Reward clipping.")

# Optimizer settings.
parser.add_argument("--learning_rate", default=0.00048,
                    type=float, metavar="LR", help="Learning rate.")
parser.add_argument("--alpha", default=0.99, type=float,
                    help="RMSProp smoothing constant.")
parser.add_argument("--momentum", default=0, type=float,
                    help="RMSProp momentum.")
parser.add_argument("--epsilon", default=0.01, type=float,
                    help="RMSProp epsilon.")
parser.add_argument("--grad_norm_clipping", default=40.0, type=float,
                    help="Global gradient norm clip.")
# yapf: enable

parser.add_argument("--hidden-dim", type=int, default=256, 
                    help="Hidden dimension")
parser.add_argument("--embedding-dim", type=int, default=5, 
                    help="object embedding dimension")
parser.add_argument("--latent-dim", type=int, default=16, 
                    help="GAN latent encoding dimension")
parser.add_argument("--secondary-embedding-dim", type=int, default=3, 
                    help="object secondary features embedding dimension")
parser.add_argument("--no-action", type=int, default=6,
                    help="index of no op action")
parser.add_argument("--vision", type=int, 
                    # default=9,
                    default=5,
                    help="range that the agent can see")
parser.add_argument("--goal", type=list, 
                    default=[8,[2,4]],
                    # default=[8,[4,8]],
                    help="goal of the agent")
parser.add_argument("--num-objects", type=int, default=11,
                    help="number of unique objects in env")
parser.add_argument("--num-colours", type=int, default=6,
                    help="number of unique colours in env")
parser.add_argument("--num-attributes", type=int, default=3,
                    help="number of unique attributes in env")
parser.add_argument('--intrinsic_reward_coef', default=1.0, type=float,
                    help='Coefficient for the intrinsic reward')
parser.add_argument("--lambda_pixel", type=float, default=25, 
                    help="pixelwise loss weight")
parser.add_argument("--lambda_latent", type=float, default=0.5, 
                    help="latent loss weight")
parser.add_argument("--lambda_kl", type=float, default=0.01, 
                    help="kullback-leibler loss weight")
parser.add_argument("--lambda_mali", type=float, default=0.005, 
                    help="mali reward loss weight")
parser.add_argument("--eps_min", type=float, default=0.05, 
                    help="min chance to sample object effects")
parser.add_argument("--eps_max", type=float, default=1., 
                    help="max chance to sample object effects")
parser.add_argument("--goal_steps_max", type=int, default=10, 
                    help="max steps with a sub-goal")
parser.add_argument("--goal_steps_min", type=int, default=2, 
                    help="min steps with a sub-goal")
parser.add_argument("--cr_scaler", type=float, default=1.5, 
                    help="scaling factor for curiousity rewards")
parser.add_argument("--npg_discount", type=float, default=0.5, 
                    help="percent of reward when primary goal is reached with another subgoal active")

logging.basicConfig(
    format=(
        "[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] " "%(message)s"
    ),
    level=0,
)




def main(flags):
    if flags.mode == "train":
        train(flags)
    else:
        test(flags)


if __name__ == "__main__":
    flags = parser.parse_args()
    main(flags)
