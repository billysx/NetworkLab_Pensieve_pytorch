"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES']='3'
import argparse
import torch
from optimizer import *
from process import local_train, local_test
import torch.multiprocessing as mp
import shutil
import torcha3c as a3c
import env
import load_trace
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(
        "a3c implement pensieve")
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.5, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=50)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=2000, help="Number of steps between savings")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=False)
    parser.add_argument("--train_traces", type=str, default="./cooked_traces/")
    parser.add_argument("--s_info", type=int, default=6)
    parser.add_argument("--s_len", type=int, default=8)
    parser.add_argument("--a_dim", type=int, default=6)
    parser.add_argument("--islstm", type=int, default=0)
    parser.add_argument('--max-grad-norm', type=float, default=10000,
                    help='clip out gradient explosion(default: 50)')
    args = parser.parse_args()
    return args


def train(args):
    print(args)
    torch.manual_seed(123)
    if os.path.isdir(args.log_path):
        shutil.rmtree(args.log_path)
    os.makedirs(args.log_path)
    if not os.path.isdir(args.saved_path):
        os.makedirs(args.saved_path)
    # Get cuda running
    mp.set_start_method('spawn')

    global_model = a3c.ActorCritic(state_dim=[args.s_info, args.s_len],
                            action_dim=args.a_dim,
                            learning_rate=[args.actor_lr, args.critic_lr],
                            islstm = args.islstm)

    global_model._initialize_weights()

    if args.use_gpu:
        global_model.cuda()
    global_model.share_memory()

    if args.load_from_previous_stage:
        if args.stage == 1:
            previous_world = args.world - 1
            previous_stage = 4
        else:
            previous_world = args.world
            previous_stage = args.stage - 1
        file_ = "{}/a3c_pensieve_{}_{}".format(args.saved_path, previous_world, previous_stage)
        if os.path.isfile(file_):
            global_model.load_state_dict(torch.load(file_))

    # optimizer = GlobalAdam([
    #     {"params":global_model.actor.parameters(), "lr":args.actor_lr},
    #     {"params":global_model.critic.parameters(), "lr":args.critic_lr}])
    # optimizer = GlobalAdam(global_model.parameters(), lr=args.actor_lr)
    actor_optimizer  = SharedAdam(global_model.actor.parameters(), lr=args.actor_lr)
    critic_optimizer = SharedAdam(global_model.critic.parameters(), lr=args.critic_lr)
    actor_optimizer.share_memory()
    critic_optimizer.share_memory()

    processes = []
    for index in range(args.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(index, args, global_model, actor_optimizer,critic_optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(index, args, global_model, actor_optimizer,critic_optimizer))
        process.start()
        processes.append(process)
    # process = mp.Process(target=local_test, args=(args.num_processes, args, global_model))
    # process.start()
    processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    args = get_args()
    train(args)