import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import numpy as np
import torch
import fixed_env as env
import a3c
import load_trace
import matplotlib.pyplot as plt
import time

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300]  # Kbps
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000
SUMMARY_DIR = './results'
LOG_FILE = './results/log_sim_rltorch'
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/pretrain_linear_reward.ckpt'

NN_MODEL = "./models/nn_model_ep_66500.ckpt"


def main():

    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    if not os.path.exists(SUMMARY_DIR):
        os.makedirs(SUMMARY_DIR)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace()

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw)

    log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
    log_file = open(log_path, 'wb')

    with torch.no_grad():
        model = a3c.ActorCritic(state_dim=[S_INFO, S_LEN],
                                action_dim=A_DIM,
                                learning_rate=[ACTOR_LR_RATE, CRITIC_LR_RATE])

        nn_model = NN_MODEL
        if nn_model is not None:  # nn_model is the path to file
            model.load_state_dict(torch.load(nn_model))
            print("Model restored.")

        time_stamp = 0

        last_bit_rate = DEFAULT_QUALITY
        bit_rate = DEFAULT_QUALITY

        action_vec = torch.zeros(A_DIM)
        action_vec[bit_rate] = 1

        s_batch = [torch.zeros(S_INFO, S_LEN)]
        a_batch = [action_vec]
        r_batch = []
        entropy_record = []

        video_count = 0

        while True:  # serve video forever
            # the action is from the last decision
            # this is to make the framework similar to the real
            delay, sleep_time, buffer_size, rebuf, \
            video_chunk_size, next_video_chunk_sizes, \
            end_of_video, video_chunk_remain = \
                net_env.get_video_chunk(bit_rate)

            time_stamp += delay  # in ms
            time_stamp += sleep_time  # in ms

            # reward is video quality - rebuffer penalty - smoothness
            reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                     - REBUF_PENALTY * rebuf \
                     - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                               VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

            r_batch.append(reward)

            last_bit_rate = bit_rate

            # log time_stamp, bit_rate, buffer_size, reward
            log_file.write((str(time_stamp / M_IN_K) + '\t' +
                           str(VIDEO_BIT_RATE[bit_rate]) + '\t' +
                           str(buffer_size) + '\t' +
                           str(rebuf) + '\t' +
                           str(video_chunk_size) + '\t' +
                           str(delay) + '\t' +
                           str(reward) + '\n').encode("utf-8"))
            log_file.flush()

            # retrieve previous state
            if len(s_batch) == 0:
                state = [np.zeros((S_INFO, S_LEN))]
            else:
                state = np.array(s_batch[-1], copy=True)

            state = torch.roll(state, -1)

            # Fill in the state vector with normalization
            state[0, -1] = torch.Tensor([VIDEO_BIT_RATE[last_bit_rate] / float(max(VIDEO_BIT_RATE))])  # last quality
            state[1, -1] = torch.Tensor([buffer_size / BUFFER_NORM_FACTOR])  # buffer size
            state[2, -1] = torch.Tensor([float(video_chunk_size) / float(delay) / M_IN_K])  # kilo byte / ms
            state[3, -1] = torch.Tensor([float(delay) / M_IN_K / BUFFER_NORM_FACTOR])  # /10 sec
            state[4, :A_DIM] = torch.Tensor([next_video_chunk_sizes]) / M_IN_K / M_IN_K  # mega byte
            # remaining chunk number
            state[5, -1] = torch.Tensor([min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)])

            logits, value = local_model(state.unsqueeze(dim=0))
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

            s_batch.append(state)
            entropy_record.append(entropy)

            if end_of_video:
                log_file.write('\n'.encode("utf-8"))
                log_file.close()

                last_bit_rate = DEFAULT_QUALITY
                bit_rate = DEFAULT_QUALITY  # use the default action here

                del s_batch[:]
                del a_batch[:]
                del r_batch[:]

                action_vec = torch.zeros(A_DIM)
                action_vec[bit_rate] = 1

                s_batch.append(torch.zeros((S_INFO, S_LEN)))
                a_batch.append(action_vec)
                entropy_record = []

                print ("video count", video_count)
                video_count += 1

                if video_count >= len(all_file_names):
                    break

                log_path = LOG_FILE + '_' + all_file_names[net_env.trace_idx]
                log_file = open(log_path, 'wb')


if __name__ == '__main__':
    main()
