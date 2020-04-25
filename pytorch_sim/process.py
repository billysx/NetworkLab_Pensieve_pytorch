"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import torch
import env
import a3c
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
import load_trace
import numpy as np

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
NUM_AGENTS = 16
TRAIN_SEQ_LEN = 100  # take as a train batch
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
HD_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log'
TEST_LOG_FOLDER = './test_results/'
TRAIN_TRACES = './cooked_traces/'
# NN_MODEL = './results/pretrain_linear_reward.ckpt'
NN_MODEL = None
print_interval = 10


def local_train(index, args, global_model, actor_optimizer, critic_optimizer, save=False):

    torch.manual_seed(614 + index)
    if save:
        start_time = timeit.default_timer()
    writer = SummaryWriter(args.log_path)

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(args.train_traces)
    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw,
                              random_seed=index)

    local_model = a3c.ActorCritic(state_dim=[S_INFO, S_LEN],
                                action_dim=A_DIM,
                                learning_rate=[ACTOR_LR_RATE, CRITIC_LR_RATE],
                                islstm = args.islstm)

    # local_model = a3c.A3C(state_dim=[S_INFO, S_LEN],
    #                             action_dim=A_DIM,
    #                             learning_rate=[ACTOR_LR_RATE, CRITIC_LR_RATE])
    local_model.train()
    local_model._initialize_weights()
    if args.use_gpu:
        local_model.cuda()


    done          = True
    curr_step     = 0
    curr_episode  = 0
    last_bit_rate = DEFAULT_QUALITY
    bit_rate      = DEFAULT_QUALITY
    time_stamp    = 0

    interval_aloss   = 0
    interval_closs   = 0
    interval_entropy = 0
    interval_reward  = []

    sum_reward   = 0
    count_reware = 0
    while True:
        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())
        state = torch.zeros(S_INFO, S_LEN)
        if done:
            cx = torch.zeros(1, 128)
            hx = torch.zeros(1, 128)
        else:
            cx = cx.detach()
            hx = hx.detach()

        if args.use_gpu:
            state = state.cuda()
        log_policies = []
        values       = []
        rewards      = []
        entropies    = []

        # One video
        while True:
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # -- linear reward --
            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
            # -- log scale reward --
            # log_bit_rate = np.log(VIDEO_BIT_RATE[bit_rate] / float(VIDEO_BIT_RATE[-1]))
            # log_last_bit_rate = np.log(VIDEO_BIT_RATE[last_bit_rate] / float(VIDEO_BIT_RATE[-1]))

            # reward = log_bit_rate \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(log_bit_rate - log_last_bit_rate)

            # -- HD reward --
            # reward = HD_REWARD[bit_rate] \
            #          - REBUF_PENALTY * rebuf \
            #          - SMOOTH_PENALTY * np.abs(HD_REWARD[bit_rate] - HD_REWARD[last_bit_rate])


            last_bit_rate = bit_rate


            state = torch.roll(state, -1)

            # Fill in the state vector with normalization
            state[0, -1] = torch.Tensor([VIDEO_BIT_RATE[last_bit_rate] / float(max(VIDEO_BIT_RATE))])  # last quality
            state[1, -1] = torch.Tensor([buffer_size / BUFFER_NORM_FACTOR])  # buffer size
            state[2, -1] = torch.Tensor([float(video_chunk_size) / float(delay) / M_IN_K])  # kilo byte / ms
            state[3, -1] = torch.Tensor([float(delay) / M_IN_K / BUFFER_NORM_FACTOR])  # /10 sec
            state[4, :A_DIM] = torch.Tensor([next_video_chunk_sizes]) / M_IN_K / M_IN_K  # mega byte
            # remaining chunk number
            state[5, -1] = torch.Tensor([min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)])
            if args.islstm == 0:
                logits, value = local_model(state.unsqueeze(dim=0))
            else:
                logits, value, hx, cx = local_model((state.unsqueeze(dim=0),hx,cx))
            # print(f"index {index}, state {state}, logits {logits}, value {value}",sep="\n")
            # print(state,logits)
            try:
                cate         = Categorical(logits)
                bit_rate     = cate.sample().item()
            except Exception as e:
                print(e)
                print(f"walking into an error of all null distribution in step {curr_step}")
                print(logits, state)
                exit()
            policy       = logits
            log_policy   = torch.log(logits)
            entropy      = (policy * log_policy).sum(1, keepdim=True)

            if curr_step > args.num_global_steps:
                done = True

            curr_step += 1
            values.append(value)
            rewards.append(reward)
            log_policies.append(log_policy[0, bit_rate])
            entropies.append(entropy)

            if end_of_video:
                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                break

        score = torch.zeros((1, 1), dtype=torch.float)
        if args.use_gpu:
            score = score.cuda()
        if not done:
            _, score = local_model(state.unsqueeze(dim=0))

        gae = torch.zeros((1, 1), dtype=torch.float)
        if args.use_gpu:
            gae = gae.cuda()
        actor_loss   = 0
        critic_loss  = 0
        entropy_loss = 0
        next_value   = score

        # for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
        #     gae = gae * args.gamma * args.tau
        #     gae = gae + reward + args.gamma * next_value.detach() - value.detach()
        #     next_value = value
        #     actor_loss = actor_loss + log_policy * gae
        #     score = score * args.gamma + reward
        #     critic_loss = critic_loss + (score - value) ** 2 / 2
        #     entropy_loss = entropy_loss + entropy

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * args.gamma * args.tau
            gae = gae + reward + args.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            score = score * args.gamma + reward
            critic_loss = critic_loss + (score - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        entropy_loss = args.beta * (entropy_loss )
        actor_loss   = -actor_loss + args.beta * entropy_loss
        # total_loss   = -actor_loss + critic_loss - entropy_loss
        writer.add_scalar("Train_{}/Loss".format(index), actor_loss, critic_loss, curr_episode)
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(global_model.parameters(), args.max_grad_norm)
        # total_loss.backward()
        # (-critic_loss).backward()
        # (actor_loss+args.beta*entropy_loss).backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                break
            global_param._grad = local_param.grad


        actor_optimizer.step()
        critic_optimizer.step()

        interval_aloss   += actor_loss.data.item()
        interval_closs   += critic_loss.data.item()
        interval_entropy += entropy_loss.data.item()
        interval_reward.append(np.sum(rewards))

        if curr_episode % print_interval == 0 :
            print("---------")
            print(f"Process {index}, episode {curr_episode}\n"+
                f"actor_loss [{interval_aloss/print_interval:4f}]  "
                f"critic_loss [{interval_closs/print_interval:4f}]  "
                f"entropy [{interval_entropy/print_interval:4f}]\n"
                f"reward [{interval_reward}]")

            if save and curr_episode % args.save_interval == 0 and curr_episode > 0:
                torch.save(global_model.state_dict(),
                           f"{args.saved_path}/a3c_{curr_episode}_reward_{sum_reward/count_reware:4f}.pkl")
            sum_reward += np.sum(interval_reward)
            count_reware += 1
            interval_aloss   = 0
            interval_closs   = 0
            interval_entropy = 0
            interval_reward  = []


        if curr_episode == int(args.num_global_steps / args.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    env, num_states, num_actions = create_train_env(args.world, args.stage, args.action_type)
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=args.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
        with torch.no_grad():
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()

        logits, value, h_0, c_0 = local_model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action)
        env.render()
        actions.append(action)
        if curr_step > args.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)