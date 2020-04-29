# Author: Bizhao Shi
# Website: https://github.com/shibizhao/Adaptive-Video-Streaming-Lab
# E-mail: shi_bizhao@pku.edu.cn
# Description: (version 1.1) 
#   A robust MPC implementation of adaptive bitrate selection to realize the maximum QoE
#   A faster and more reasonable implementation of MPC method.
# Update:
#   1. Remove the video_size arrays, the future chunk information is only the 'nextVideoChunkSize'
#   2. Remove some jitter solutions from 'chunkOptionsSet' using Hamming Distance
#   3. Edit the 'localReward' update logic


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


# global variables defination
chunkOptionsSet = []
pastErrors = []
pastBWEsts = []

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

    # enumerate all possible solutions of future chunks
    for solution in itertools.product([i for i in range(bitRatesTypes)],
                                     repeat=defaultFutureChunkCount):
        chunkOptionsSet.append(solution)
    
    # remove some jitter solutions 
    for solution in chunkOptionsSet:
        hammingDistance = 0
        for pos in range(1, defaultFutureChunkCount):
            hammingDistance += abs(solution[pos] - solution[pos-1])
        if hammingDistance > 7:
            chunkOptionsSet.remove(solution)
    
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
        pastBWEsts.append(currentPredBW) 

        # get the video chunks information of this round prediction
        currentLastIndex = totalChunksCount - chunkRemainCount
        currentFutureChunkCount = min(chunkRemainCount, defaultFutureChunkCount)

        # enumerate all the possible solutions and pick the best one
        bestReward = -INF
        bestSolution = ()
        finalOption = -1
        startBufferSize = currentBufferSize
        startTime = time.time()

        # use the predicted bandwidth and the next chunk size to calculate the estimated download time
        allDownloadTime = []
        for option in range(0, bitRatesTypes):
            allDownloadTime.append((float(nextVideoChunkSize[option]) / (bitsFactor * bitsFactor))/ currentPredBW)

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
                downloadTime =  allDownloadTime[thisChunkOption]
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
            # update the information
            if localReward >= bestReward:
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

            outputFileName = outputFilePrefix + "_optim_" + allFileNames[netEnvironment.trace_idx]
            outputFilePointer = open(outputFileName, "wb")

if __name__ == '__main__':
    st = time.time()
    main()
    print(time.time() - st)
        

            

        





