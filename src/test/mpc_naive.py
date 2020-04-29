# Author: Bizhao Shi
# Website: https://github.com/shibizhao/Adaptive-Video-Streaming-Lab
# E-mail: shi_bizhao@pku.edu.cn
# Description: (version 1.0) 
#   A robust MPC implementation of adaptive bitrate selection to realize the maximum QoE
#   A naive code migration from mpc.py
# Bug Fixed:
#   1. the list 'pastBWEsts' will append the 'currentPredBW', rather than 'harmonicBW'
#   2. the 'nextVideoChunkSize' is added as one of the return values from "get_video_chunk"


import numpy as np
import fixed_env as env
import load_trace
import itertools
import time

# constant defination
INF = 100000000000

stateInfoLength = 5
pastFramesLength = 8
bitRatesTypes = 6
bitRatesOptions = [300, 750, 1200, 1850, 2850, 4300]
minBitRate = np.min(bitRatesOptions)
maxBitRate = np.max(bitRatesOptions)


defaultFutureChunkCount = 5
defaultChunkCountToEnd = 48.0
totalChunksCount = 48

millsecondsPerSecond = 1000.0

bitsFactor = 1000.0

rebufferFactor = 4.3
smoothFactor = 1.0
bufferNormFactor = 10.0

defaultBitRateOption = 1

outputDirctory = "./results"
outputFilePrefix = outputDirctory + '/log_sim_mpc'

chunkOptionsSet = []

# global variables defination
pastErrors = []
pastBWEsts = []

size_video1 = [2354772, 2123065, 2177073, 2160877, 2233056, 1941625, 2157535, 2290172, 2055469, 2169201, 2173522, 2102452, 2209463, 2275376, 2005399, 2152483, 2289689, 2059512, 2220726, 2156729, 2039773, 2176469, 2221506, 2044075, 2186790, 2105231, 2395588, 1972048, 2134614, 2164140, 2113193, 2147852, 2191074, 2286761, 2307787, 2143948, 1919781, 2147467, 2133870, 2146120, 2108491, 2184571, 2121928, 2219102, 2124950, 2246506, 1961140, 2155012, 1433658]
size_video2 = [1728879, 1431809, 1300868, 1520281, 1472558, 1224260, 1388403, 1638769, 1348011, 1429765, 1354548, 1519951, 1422919, 1578343, 1231445, 1471065, 1491626, 1358801, 1537156, 1336050, 1415116, 1468126, 1505760, 1323990, 1383735, 1480464, 1547572, 1141971, 1498470, 1561263, 1341201, 1497683, 1358081, 1587293, 1492672, 1439896, 1139291, 1499009, 1427478, 1402287, 1339500, 1527299, 1343002, 1587250, 1464921, 1483527, 1231456, 1364537, 889412]
size_video3 = [1034108, 957685, 877771, 933276, 996749, 801058, 905515, 1060487, 852833, 913888, 939819, 917428, 946851, 1036454, 821631, 923170, 966699, 885714, 987708, 923755, 891604, 955231, 968026, 874175, 897976, 905935, 1076599, 758197, 972798, 975811, 873429, 954453, 885062, 1035329, 1026056, 943942, 728962, 938587, 908665, 930577, 858450, 1025005, 886255, 973972, 958994, 982064, 830730, 846370, 598850]
size_video4 = [668286, 611087, 571051, 617681, 652874, 520315, 561791, 709534, 584846, 560821, 607410, 594078, 624282, 687371, 526950, 587876, 617242, 581493, 639204, 586839, 601738, 616206, 656471, 536667, 587236, 590335, 696376, 487160, 622896, 641447, 570392, 620283, 584349, 670129, 690253, 598727, 487812, 575591, 605884, 587506, 566904, 641452, 599477, 634861, 630203, 638661, 538612, 550906, 391450]
size_video5 = [450283, 398865, 350812, 382355, 411561, 318564, 352642, 437162, 374758, 362795, 353220, 405134, 386351, 434409, 337059, 366214, 360831, 372963, 405596, 350713, 386472, 399894, 401853, 343800, 359903, 379700, 425781, 277716, 400396, 400508, 358218, 400322, 369834, 412837, 401088, 365161, 321064, 361565, 378327, 390680, 345516, 384505, 372093, 438281, 398987, 393804, 331053, 314107, 255954]
size_video6 = [181801, 155580, 139857, 155432, 163442, 126289, 153295, 173849, 150710, 139105, 141840, 156148, 160746, 179801, 140051, 138313, 143509, 150616, 165384, 140881, 157671, 157812, 163927, 137654, 146754, 153938, 181901, 111155, 153605, 149029, 157421, 157488, 143881, 163444, 179328, 159914, 131610, 124011, 144254, 149991, 147968, 161857, 145210, 172312, 167025, 160064, 137507, 118421, 112270]

def getChunkSize(quality, index):
    if ( index < 0 or index > 48 ):
        return 0
    # note that the quality and video labels are inverted (i.e., quality 4 is highest and this pertains to video1)
    sizes = {5: size_video1[index], 4: size_video2[index], 3: size_video3[index], 2: size_video4[index], 1: size_video5[index], 0:size_video6[index]}
    return sizes[quality]


def main():
    # check the constant defination is valid or not
    assert len(bitRatesOptions) == bitRatesTypes
    
    # load the traces
    allCookedTime, allCookedBW, allFileNames = load_trace.load_trace()

    # set the environment
    netEnvironment = env.Environment(all_cooked_time=allCookedTime,
                                      all_cooked_bw=allCookedBW)

    # open the output log file to write
    outputFileName = outputFilePrefix + "_" + allFileNames[netEnvironment.trace_idx]
    outputFilePointer = open(outputFileName, "wb")

    # initial the local variables
    timeStamp = 0
    lastBitRateOption = defaultBitRateOption
    currentBitRateOption = defaultBitRateOption
    videoCount = 0
    historyState = np.zeros((stateInfoLength, pastFramesLength))

    # enum all possible solutions of future chunks
    for solution in itertools.product([i for i in range(bitRatesTypes)],
                                     repeat=defaultFutureChunkCount):
        chunkOptionsSet.append(solution)

    # computing kernel:
    while True:
        # get the video chunk according to the current bitrate option
        assert currentBitRateOption >= 0
        delay, sleepTime, currentBufferSize, rebuffer, currentVideoChunkSize, \
            nextVideoChunkSize, endFlag, chunkRemainCount = netEnvironment.get_video_chunk(currentBitRateOption)
        
        # update the time stamp because of the delay and sleeping time
        timeStamp += delay + sleepTime  # ms

        # calculate the reward value according to the formula
        qualityValue = bitRatesOptions[currentBitRateOption] / bitsFactor # kb to Mb
        smoothValue = np.abs(bitRatesOptions[currentBitRateOption] \
                    - bitRatesOptions[lastBitRateOption]) / bitsFactor
        rewardValue =  qualityValue \
                    - rebufferFactor * rebuffer \
                    - smoothFactor * smoothValue
        
        # write the output file
        outputItemStr = str(timeStamp / millsecondsPerSecond) + '\t' \
                    + str(bitRatesOptions[currentBitRateOption]) + '\t' \
                    + str(currentBufferSize) + '\t' \
                    + str(rebuffer) + '\t' \
                    + str(currentVideoChunkSize) + '\t' \
                    + str(delay) + '\t' \
                    + str(rewardValue) + '\n'
        outputFilePointer.write(outputItemStr.encode('utf-8'))
        outputFilePointer.flush()

        # update the bit rate option
        lastBitRateOption = currentBitRateOption

        # update the history state information like a sliding window
        historyState = np.roll(historyState, -1, axis=1)
        historyState[0, -1] = bitRatesOptions[currentBitRateOption] / float(maxBitRate)
        historyState[1, -1] = currentBufferSize / bufferNormFactor
        historyState[2, -1] = rebuffer
        historyState[3, -1] = float(currentVideoChunkSize) / float(delay) / bitsFactor
        historyState[4, -1] = np.minimum(chunkRemainCount, defaultChunkCountToEnd) / float(defaultChunkCountToEnd)

        # MPC kernel begin
        # calculate the normaliztion estimated error of bandwidth
        currentError = 0.
        if(len(pastBWEsts) > 0):
            currentError = abs(pastBWEsts[-1] - historyState[3, -1]) / float(historyState[3, -1])
        pastErrors.append(currentError)
        
        # calculate the harmonic mean of last 5 history bandwidths
        # Step 1: collect the last 5 history bandwidths
        pastRealBWArray = historyState[3, -5:]
        while pastRealBWArray[0] == 0.0:
            pastRealBWArray = pastRealBWArray[1:]
        
        # Step 2: calculate the harmonic mean
        pastRealBWSum = 0.0
        for pastRealBWItems in pastRealBWArray:
            pastRealBWSum += (1 / float(pastRealBWItems))
        harmonicBW = 1.0 / (pastRealBWSum / len(pastRealBWArray))

        # calculate the predicted future bandwidth according to the est. error and harmonic mean
        errorIndex = min(5, len(pastErrors))
        maxError = float(max(pastErrors[-errorIndex:]))
        currentPredBW = harmonicBW / (1 + maxError)
        pastBWEsts.append(currentPredBW)    # fixed this bug, reward increases

        # get the video chunks information of this round prediction
        currentLastIndex = totalChunksCount - chunkRemainCount
        currentFutureChunkCount = min(chunkRemainCount, defaultFutureChunkCount)

        # enumerate all the possible solutions and pick the best one
        bestReward = -INF
        bestSolution = ()
        finalOption = -1
        startBufferSize = currentBufferSize

        for solution in chunkOptionsSet:
            localSolution = solution[0:currentFutureChunkCount]
            localRebufferTime = 0.0
            localCurrentBufferSize = startBufferSize
            localBitRateSum = 0.
            localSmoothDiffs = 0.
            localLastChunkOption = currentBitRateOption
            # the 5 future chunks loop
            for pos in range(0, currentFutureChunkCount):
                thisChunkOption = localSolution[pos]
                thisIndex = currentLastIndex + pos + 1
                thisChunkSize = getChunkSize(thisChunkOption, thisIndex)
                downloadTime = (float(thisChunkSize) / (bitsFactor * bitsFactor) ) / currentPredBW # Bytes to MBytes
                if localCurrentBufferSize < downloadTime:
                    localRebufferTime += downloadTime - localCurrentBufferSize
                    localCurrentBufferSize = 0
                else:
                    localCurrentBufferSize -= downloadTime
                # This 4 means the play speed
                localCurrentBufferSize += 4
                localBitRateSum += bitRatesOptions[thisChunkOption]
                localSmoothDiffs += abs(bitRatesOptions[thisChunkOption] - bitRatesOptions[localLastChunkOption])
                localLastChunkOption = thisChunkOption
            
            localReward = float(localBitRateSum) / bitsFactor \
                             - rebufferFactor * localRebufferTime \
                             - float(localSmoothDiffs) / bitsFactor
            if localReward >= bestReward:
                if bestSolution != () and bestSolution[0] < localSolution[0]:
                    bestSolution = localSolution
                else:
                    bestSolution = localSolution
                bestReward = localReward
                if bestSolution != ():
                    finalOption = bestSolution[0]
        currentBitRateOption = finalOption
        
        if endFlag:
            outputFilePointer.write("\n".encode('utf-8'))
            outputFilePointer.close()

            lastBitRateOption = defaultBitRateOption
            currentBitRateOption = defaultBitRateOption
            historyState = np.zeros((stateInfoLength, pastFramesLength))

            print("video count", videoCount)
            videoCount += 1

            if videoCount >= len(allFileNames):
                break

            outputFileName = outputFilePrefix + "_naive_" + allFileNames[netEnvironment.trace_idx]
            outputFilePointer = open(outputFileName, "wb")

if __name__ == '__main__':
    st = time.time()
    main()
    print(time.time() - st)

        

            

        





