import sys
import os
import numpy as np
import theano
import theano.tensor as T
import time
import datetime

utilsPath = '../utils'
sys.path.append(os.path.abspath(utilsPath))
import imgSource
import vis


currentLocalTimeString = lambda :datetime.datetime.fromtimestamp(time.time()).strftime('%H:%M:%S')


def makeGradFn():
    W = T.ftensor4('W') # The filters to learn. L2 norm should be one.
    V = T.ftensor4('V') # Principle Component to remove. L2 norm must be one. Must be orthonormal.
    X = T.ftensor4('X') # Input Images.
    B = T.ftensor4('B') # Input bias, must be of shape(L, 1, 1, 1).
    Epsilon = T.fscalar('Epsilon') # Smoothing factor. Should be positive and small.

    convResult = T.nnet.conv2d(X, W) - T.transpose(T.tensordot(T.nnet.conv2d(X, V), T.tensordot(W, V, axes=((1,2,3),(1,2,3))), axes=((1,),(1,))), axes=(0,3,1,2))
    preActi = convResult + T.addbroadcast(T.transpose(B, axes=(1, 0, 2, 3)), 0, 2, 3)
    WindowScores = T.sum(T.nnet.relu(preActi), axis=1) + Epsilon
    Loss = T.mean(-T.log(WindowScores))

    W_grad = T.grad(Loss, W)
    W_update = - W_grad / T.addbroadcast(T.sqrt(T.sum(W_grad * W_grad, axis=(1, 2, 3), keepdims=True)), 1, 2, 3)

    return theano.function([X, W, B, V, Epsilon], [Loss, W_update])


def batchOpti(imageData, B_val, V_val, batchSize, filterSize, filterNum, convThres=0.999, maxIter=20, epsilon=1e-6):
    inputChannelNum = imageData.shape[1]

    W_val = np.random.normal(size=(filterNum, inputChannelNum, filterSize, filterSize)).astype('float32')
    W_val /= np.sqrt(np.sum(W_val * W_val, axis=(1, 2, 3), keepdims=True))

    numBatch = imageData.shape[0] // batchSize
    assert numBatch * batchSize == imageData.shape[0]

    imageMean = np.mean(imageData, axis=0)
    imageData -= imageMean

    gradFn = makeGradFn()

    lastCosSim = 0.0
    cnt = 0
    while lastCosSim < convThres and cnt < maxIter:
        W_update = np.zeros(W_val.shape, dtype=np.float32)
        loss = 0.0
        for j in xrange(numBatch):
            batchLoss, batchUpdate = gradFn(imageData[j * batchSize:(j+1)*batchSize, :, :, :], W_val, B_val, V_val, epsilon)
            loss += batchLoss
            W_update += batchUpdate
        W_update /= np.sqrt(np.sum(W_update * W_update, axis=(1, 2, 3), keepdims=True))
        lastCosSim = np.mean(np.sum(W_val * W_update, axis=(1, 2, 3)))
        print currentLocalTimeString(), 'loss', loss / numBatch, 'change', lastCosSim 
        W_val = W_update
        cnt += 1
    print 'Finish optimization in', cnt, 'epochs.'
    return W_val, imageMean


def simple1(b, makeGray=False, imageNum=1000, batchSize=20, filterSize=11, filterNum=36):
    imageData = imgSource.getImage256Tensor(imageNum)
    if makeGray:
        imageData = imageData.mean(axis=1, keepdims=True)

    V_val = np.ones((1, imageData.shape[1], filterSize, filterSize), dtype=np.float32)
    V_val /= np.sqrt(np.sum(V_val * V_val, axis=(1, 2, 3), keepdims=True))

    B_val = np.ones((filterNum, 1, 1, 1), dtype=np.float32) * b

    W_val, imageMean = batchOpti(imageData, B_val, V_val, batchSize, filterSize, filterNum)

    return np.transpose(W_val, axes=(0,2,3,1))


def main():
    vis.visFilters(simple1(0.0, makeGray=False), 'simple1_C00.png')
    vis.visFilters(simple1(-0.2, makeGray=False), 'simple1_C02.png')
    vis.visFilters(simple1(-0.4, makeGray=False), 'simple1_C04.png')

    vis.visFilters(simple1(0.0, makeGray=True), 'simple1_G00.png')
    vis.visFilters(simple1(-0.2, makeGray=True), 'simple1_G02.png')
    vis.visFilters(simple1(-0.4, makeGray=True), 'simple1_G04.png')


if __name__ == '__main__':
    sys.exit(int(main() or 0))
