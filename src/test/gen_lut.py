# Author: Bizhao Shi
# Website: https://github.com/shibizhao/Adaptive-Video-Streaming-Lab
# E-mail: shi_bizhao@pku.edu.cn
# Description:  
#   Produce the offline LUT for the Fast MPC.
# Notes:
#   This implementation considers the buffer size level, current bitrate option,
#   and the minimum bitrate download time of the next video chunk.
#   I found that the ChunkSize is proportional to Bitrate Value.
#   Different from the FMPC paper.

import numpy as np
import itertools
import time

# constant defination

INF = 100000000000


bitRatesTypes = 6
bitRatesOptions = [300, 750, 1200, 1850, 2850, 4300]
defaultFutureChunkCount = 5

millsecondsPerSecond = 1000.0

bitsFactor = 1000.0

rebufferFactor = 4.3
smoothFactor = 1.0
bufferNormFactor = 10.0

defaultBitRateOption = 1

chunkOptionsSet = []

# Further LUT Compression using the vectorization
isVector = False

# enum all possible solutions of future chunks
for solution in itertools.product([i for i in range(bitRatesTypes)],
                                     repeat=defaultFutureChunkCount):
    chunkOptionsSet.append(solution)

# produce the bufferSizeLevel list
bufferSizeStart = 1
bufferSizeEnd = 31
bufferSizeStride = 1.0
bufferSizeLevelList = [float(i * bufferSizeStride) for i in range(bufferSizeStart, bufferSizeEnd)]

# produce the download time of 300kbps
downloadTimeStart = 1
downloadTimeEnd = 101
downloadTimeStride = 0.04
downloadTimeBaseLineList = [float(i * downloadTimeStride) for i in range(downloadTimeStart, downloadTimeEnd)]

# produce the current options list
nowOptionList = [i for i in range(0, bitRatesTypes)] 

vecString = ""
def main():
    if isVector:
        enumFP = open("./enum_vec.txt", "w", encoding="utf-8", errors='ignored')
    else:
        enumFP = open("./enum.txt", "w", encoding="utf-8", errors='ignored')
    for bfSize in bufferSizeLevelList:
        print("log: %.1f start" % bfSize)
        for dlTime in downloadTimeBaseLineList:
            currentAllDownLoadTime = []
            currentAllDownLoadTime.append(dlTime)
            for j in range(1, bitRatesTypes):
                currentAllDownLoadTime.append((float(dlTime)/bitRatesOptions[0]) * bitRatesOptions[j])
            for op in nowOptionList:
                bestReward = -INF
                bestSolution = ()
                finalOption = -1
                for solution in chunkOptionsSet:
                    localSolution = solution[0:defaultFutureChunkCount]
                    localRebufferTime = 0.0
                    localCurrentBufferSize = bfSize
                    localBitRateSum = 0.
                    localSmoothDiffs = 0.
                    localLastChunkOption = op
                    for pos in range(0, defaultFutureChunkCount):
                        thisChunkOption = localSolution[pos]
                        downloadTime = currentAllDownLoadTime[thisChunkOption]
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
                        bestSolution = localSolution
                        bestReward = localReward
                        if bestSolution != ():
                            finalOption = bestSolution[0]
                assert finalOption >= 0
                if isVector:
                    vecString += str(int(finalOption)) + ' '                    
                else:
                    print("%.1f" % bfSize, "%.2f" % dlTime, "%d" % op, "%d" % finalOption, file=enumFP)

    if isVector:
        print(vecString, file=enumFP)
    enumFP.close()

if __name__ == '__main__':
    main()

        

            

        





