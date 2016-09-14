import sys
import os
import numpy as np

utilsPath = '../utils'
sys.path.append(os.path.abspath(utilsPath))
import imgSource
import vis


def findNextPC(data, convThres):
    w = np.random.normal(size=(data.shape[1],))
    w = w / np.sqrt(np.sum(w * w))
    lastCosSim = 0
    cnt = 0
    while lastCosSim < convThres:
        w_update = np.sum(np.sum(data * w, axis=1, keepdims=True) * data, axis=0)
        w_update = w_update / np.sqrt(np.sum(w_update * w_update))
        lastCosSim = np.sum(w * w_update)
        w = w_update
        cnt += 1
    print 'Found next PC, using', cnt, 'iter'
    return w


def removePCfromData(data, w):
    return data - np.sum(data * w, axis=1, keepdims=True) * w


def pca_fast(data, n, convThres):
    data = np.copy(data - np.mean(data, axis=0))
    pc = np.zeros((n, data.shape[1]), dtype=np.float32)
    for i in xrange(n):
        w = findNextPC(data, convThres)
        data = removePCfromData(data, w)
        pc[i, :] = w
    return pc


def findColorPCs(n, filterSize, convThres=0.999, kSamp=100000):
    imgD = imgSource.randColorCorp(kSamp, filterSize)
    imgD = np.reshape(imgD, (kSamp, filterSize*filterSize*3))
    return pca_fast(imgD, n, convThres).reshape(n, filterSize, filterSize, 3)


def findGrayPCs(n, filterSize, convThres=0.999, kSamp=100000):
    imgD = imgSource.randColorCorp(kSamp, filterSize)
    imgD = np.mean(imgD, axis=3)
    imgD = np.reshape(imgD, (kSamp, filterSize*filterSize))
    return pca_fast(imgD, n, convThres).reshape(n, filterSize, filterSize)


def main():
    vis.visFilters(findColorPCs(36, 11), 'ColorPCA.png')
    vis.visFilters(findGrayPCs(36, 11), 'GrayPCA.png')

if __name__ == '__main__':
    sys.exit(int(main() or 0))
