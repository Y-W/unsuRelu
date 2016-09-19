import sys, os, cPickle, gzip
import numpy as np
from sklearn import svm

utilsPath = '../utils'
sys.path.append(os.path.abspath(utilsPath))
import vis

with gzip.open('../../mnist/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = cPickle.load(f)

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


def findMnistPCA(n, convThres=0.999):
    imgD = train_set[0]
    return pca_fast(imgD, n, convThres)


def removeAllPCs(data, W):
    data = np.copy(data - np.mean(data, axis=0))
    for i in xrange(W.shape[0]):
        data = removePCfromData(data, W[i, :])
    return data


def unsuReluUpdateWithReset(X, W, bias, epsilon=1e-6):
    # X: n * d; W: k * d
    activation = np.maximum(np.dot(X, W.T) + bias, 0)
    scores = np.sum(activation, axis=1) + epsilon
    sampleWeights = 1.0 / scores
    newW = np.zeros(W.shape, dtype=np.float32)
    resetCnt = 0
    for i in xrange(W.shape[0]):
        negGrad = np.dot((activation[:, i]>0) * sampleWeights, X)
        if np.all(negGrad == 0):
            negGrad = np.random.normal(size=(W.shape[1],)).astype(np.float32)
            resetCnt += 1
        newW[i, :] = negGrad / np.linalg.norm(negGrad)
    print 'loss:', np.mean(-np.log(scores)),'Reseted: ', resetCnt
    return newW


def unsuReluWithReset(X, k, bias, threshold=0.999):
    X = np.copy(X / np.sqrt(np.sum(X * X, axis=1, keepdims=True)))
    W = np.random.normal(size=(k, X.shape[1])).astype(np.float32)
    W /= np.sqrt(np.sum(W*W, axis=1, keepdims=True))
    lastSim = 0.0
    while lastSim < threshold:
        newW = unsuReluUpdateWithReset(X, W, bias)
        lastSim = np.mean(np.sum(newW * W, axis=1))
        W = newW
        print lastSim
    return newW


def main():
    PCs = findMnistPCA(10)
    vis.visFilters(PCs.reshape(10, 28, 28), 'mnistPCA.png')
    kPC = 0
    newTrainSet = removeAllPCs(train_set[0], PCs[:kPC, :])
    bias = -0.15
    W = unsuReluWithReset(newTrainSet, 100, bias)
    vis.visFilters(W.reshape(W.shape[0], 28, 28), 'mnistFilters.png')

    processData = lambda input: np.maximum(np.dot(removeAllPCs(input, PCs[:kPC, :]), W.T) + bias, 0)

    linSVM = svm.LinearSVC()
    linSVM.fit(processData(train_set[0]), train_set[1])
    print linSVM.score(processData(train_set[0]), train_set[1])
    print linSVM.score(processData(valid_set[0]), valid_set[1])

    linSVM2 = svm.LinearSVC()
    linSVM2.fit(train_set[0], train_set[1])
    print linSVM2.score(train_set[0], train_set[1])
    print linSVM2.score(valid_set[0], valid_set[1])



if __name__ == '__main__':
    sys.exit(int(main() or 0))
