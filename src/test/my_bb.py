import numpy as np
import fixed_env_mpdash as env
import load_trace

# video player settings
VIDEO_BITRATE = [300,750,1200,1850,2850,4300]
TOT_BITRATE_COUNT = 6
DEFAULT_BITRATE_INDEX = 1
CHUNK_LEN = 4
MAX_BUFFER_SIZE = 60
# QoE settings
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1
# algorithm settings
RESERVOIR = 9
CUSHION = 22
RANDOM_SEED = 42
MPD_ENABLE_LO = RESERVOIR + CHUNK_LEN           # threshold for mpdash to close
MPD_ENABLE_HI = MAX_BUFFER_SIZE - CHUNK_LEN     # threshold for mpdash to open
ESTIMATION = 0.95
CELLULAR_ENABLED = True
MPDASH_ENABLE = True
# log file settings
DIR = './results'
LOG_FILE_HDR = './results/log_sim_my_bb'

def ratemap(buffer_size):
    choose = float(VIDEO_BITRATE[0]) + (float(buffer_size) - float(RESERVOIR)) * \
            (float(VIDEO_BITRATE[TOT_BITRATE_COUNT-1]) - float(VIDEO_BITRATE[0])) / float(CUSHION)
    min_distance = 100000
    choose_index = 0
    for i in range(TOT_BITRATE_COUNT):    # find the nearest rate
        if(abs(choose - VIDEO_BITRATE[i]) < min_distance):
            min_distance = abs(choose-VIDEO_BITRATE[i])
            choose_index = i
    return int(choose_index)

def bba(buffer_size,last_bit_rate,next_bit_rate,video_chunk_size,send_bytes,ddl_window,delay):
    # buffer based algorithm
    if buffer_size < RESERVOIR:
        bit_rate = 0
    elif buffer_size >= RESERVOIR + CUSHION:
        bit_rate = TOT_BITRATE_COUNT - 1
    else:
        choice = 1          # if set to 1, algorithm is based on ratemap, if default, adds future bandwidth estimation
        next_iteration = 1  # if set to 1, future bandwidth estimation uses iteration, if default, uses the average of the past
        if(choice == 1): 
            bit_rate = ratemap(buffer_size)
            default_bit_rate = int((TOT_BITRATE_COUNT - 1) * (buffer_size - RESERVOIR) / float(CUSHION)) 
        else:
            if(next_iteration == 1):
                next_buffer_size = np.maximum(buffer_size - delay / M_IN_K, 0.0)
                next_buffer_size += CHUNK_LEN
                next_bit_rate = int(ratemap(next_buffer_size))
                if(next_bit_rate < 0):
                    next_bit_rate = 0
                if(next_bit_rate >= TOT_BITRATE_COUNT):
                    next_bit_rate = TOT_BITRATE_COUNT - 1
            min_distance = 100000
            for i in range(TOT_BITRATE_COUNT):
                if(abs(next_bit_rate - VIDEO_BITRATE[i]) < min_distance):
                    min_distance = abs(next_bit_rate-VIDEO_BITRATE[i])
                    next_nearest = i
            if(last_bit_rate == next_nearest):
                bit_rate = next_nearest
            else:
                bit_rate = ratemap(buffer_size)

    # MP-DASH adapter
    if(buffer_size < MPD_ENABLE_LO):
        MPDASH_ENABLE = False
    elif(buffer_size > MPD_ENABLE_HI):
        ddl_window += (buffer_size - MPD_ENABLE_HI)
        MPDASH_ENABLE = True
    else:
        MPDASH_ENABLE = True
    # MP_DASH scheduler
    if(MPDASH_ENABLE):
        estimated_throughput = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        if(((ESTIMATION * ddl_window - delay) * estimated_throughput) > (video_chunk_size - send_bytes)):
            CELLULAR_ENABLED = False
        else:
            CELLULAR_ENABLED = True

    return int(bit_rate)

def main():

    np.random.seed(RANDOM_SEED)
    TOT_BITRATE_COUNT = len(VIDEO_BITRATE)

    traces_time, traces_bandwidth, file_names = load_trace.load_trace()
    net_env = env.Environment(all_cooked_time = traces_time, all_cooked_bw = traces_bandwidth)

    log_filename = LOG_FILE_HDR + '_' + file_names[net_env.trace_idx]
    log_file = open(log_filename, 'wb')

    video_count = 0
    time_stamp = 0
    last_bit_rate = DEFAULT_BITRATE_INDEX
    bit_rate = DEFAULT_BITRATE_INDEX
    ddl_window = CHUNK_LEN
    history_bitrate = []

    while True:
        delay,                                           \
        sleep_time,                                      \
        buffer_size,                                     \
        rebuf,                                           \
        video_chunk_size,                                \
        next_video_chunk_sizes,                          \
        send_bytes,                                      \
        end_of_video,                                    \
        video_chunk_remain=                              \
        net_env.get_video_chunk(bit_rate,CELLULAR_ENABLED)

        history_bitrate.append(VIDEO_BITRATE[bit_rate])
        time_stamp = time_stamp + delay + sleep_time
        
        reward = VIDEO_BITRATE[bit_rate] / M_IN_K               \
               - REBUF_PENALTY * rebuf                          \
               - SMOOTH_PENALTY * abs(VIDEO_BITRATE[bit_rate] - \
                                      VIDEO_BITRATE[last_bit_rate]) / M_IN_K

        log_file_write = (str(time_stamp / M_IN_K)    + '\t' +
                         str(VIDEO_BITRATE[bit_rate]) + '\t' +
                         str(buffer_size)             + '\t' +
                         str(rebuf)                   + '\t' +
                         str(video_chunk_size)        + '\t' +
                         str(delay)                   + '\t' +
                         str(reward)                  + '\n')
        log_file.write(log_file_write.encode('ANSI'))
        log_file.flush()

        last_bit_rate = bit_rate
        next_bit_rate = np.mean(history_bitrate)
        bit_rate = bba(buffer_size,last_bit_rate,next_bit_rate,video_chunk_size,send_bytes,ddl_window,delay)

        if end_of_video:
            log_file.write('\n'.encode('ANSI'))
            log_file.close()

            last_bit_rate = DEFAULT_BITRATE_INDEX
            bit_rate = DEFAULT_BITRATE_INDEX  # use the default action here
            ddl_window = CHUNK_LEN
            history_bitrate = []

            print("video count=", video_count,"file=", log_filename)
            video_count += 1
            if (video_count >= len(file_names)):
                break
            
            log_filename = LOG_FILE_HDR + '_' + file_names[net_env.trace_idx]
            log_file = open(log_filename, 'wb')
     

if __name__ == '__main__':
    main()
