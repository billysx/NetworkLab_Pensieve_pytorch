# Author: Bizhao Shi
# Website: https://github.com/shibizhao/Adaptive-Video-Streaming-Lab
# E-mail: shi_bizhao@pku.edu.cn
# Description: (version 2.1) 
#   A robust MPC implementation of adaptive bitrate selection to realize the maximum QoE
#   An implementation of FastMPC method using a naive Look-up Table.
# Update:
#   1. Discrete the real value to the preset value and make a decision via LUT.

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

# load enum look up table

lookupTable = []

def initialLookUpTable():
    enumFP = open("enum.txt",'r',encoding='utf-8',errors='ignored')
    while True:
        content = enumFP.readline()
        if content:
            contentArray = content.split(' ')
            assert len(contentArray) == 4
            lookupTable.append((float(contentArray[0]), float(contentArray[1]),int(contentArray[2]),int(contentArray[3])))
        else:
            break
    enumFP.close()

# According to the current buffer size, download time and bitrate option to make a decision
def Decision(bfSize, dlTime, curOp):
    
    flt1d = round(bfSize, 0)
    flt1d = min(flt1d, 30.0)
    fltid = max(flt1d, 1.0)
    flt2d = round(dlTime / 0.04, 0) * 0.04
    flt2d = min(flt2d, 4.00)
    flt2d = max(flt2d, 0.04)
    int3d = int(curOp)
    index1d = int(flt1d / 1.0) - 1
    index2d = int(flt2d / 0.04) - 1
    indexAll = index1d * 600 + index2d * 6 + int3d
    return lookupTable[indexAll][3]



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

    # initial the look up table
    initialLookUpTable()
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
        pastBWEsts.append(currentPredBW)    # fixed this bug, reward increases 1.xx


        # use the predicted bandwidth and the next chunk size to calculate the estimated download time
        allDownloadTime = []
        for option in range(0, bitRatesTypes):
            allDownloadTime.append((float(nextVideoChunkSize[option]) / (bitsFactor * bitsFactor))/ currentPredBW)

        finalOption = Decision(currentBufferSize, allDownloadTime[0], currentBitRateOption)
        currentBitRateOption = finalOption

        assert finalOption >= 0
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

            outputFileName = outputFilePrefix + "_lut_" + allFileNames[netEnvironment.trace_idx]
            outputFilePointer = open(outputFileName, "wb")

if __name__ == '__main__':
    st = time.time()
    main()
    print(time.time() - st)

        

            

        





