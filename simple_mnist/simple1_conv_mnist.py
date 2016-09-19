import sys
import os
import numpy as np
import theano
import theano.tensor as T
import time
import datetime
from sklearn import linear_model

utilsPath = '../utils'
sys.path.append(os.path.abspath(utilsPath))
import vis

import pickle, gzip
with gzip.open('../../mnist/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)


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


def makeActivFn():
    W = T.ftensor4('W') # The filters to learn. L2 norm should be one.
    X = T.ftensor4('X') # Input Images.
    B = T.ftensor4('B') # Input bias, must be of shape(L, 1, 1, 1).

    convResult = T.nnet.conv2d(X, W)
    preActi = convResult + T.addbroadcast(T.transpose(B, axes=(1, 0, 2, 3)), 0, 2, 3)
    acti = T.nnet.relu(preActi)
    return theano.function([X, W, B], [acti])


def batchOpti(imageData, B_val, V_val, batchSize, filterSize, filterNum, convThres=0.999, maxIter=50, epsilon=1e-6):
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
        normTmp = np.sum(W_update * W_update, axis=(1, 2, 3))
        for j in xrange(filterNum):
            if normTmp[j] == 0:
                W_update[j, :, :, :] = np.random.normal(size=(inputChannelNum, filterSize, filterSize)).astype('float32')
        W_update /= np.sqrt(np.sum(W_update * W_update, axis=(1, 2, 3), keepdims=True))
        lastCosSim = np.mean(np.sum(W_val * W_update, axis=(1, 2, 3)))
        print currentLocalTimeString(), 'loss', loss / numBatch, 'change', lastCosSim 
        W_val = W_update
        cnt += 1
    print 'Finish optimization in', cnt, 'epochs.'
    return W_val, imageMean


def pipeData(inputSet, imageMean, filters, bias, imageSize=28):
    actiFn = makeActivFn()
    inputSet = inputSet.reshape((-1, 1, imageSize, imageSize)).astype(np.float32)
    inputSet -= imageMean
    B = np.ones((filters.shape[0], 1, 1, 1), dtype=np.float32) * bias
    (acti,) = actiFn(inputSet, filters, B)
    return acti.reshape((inputSet.shape[0], -1))


def simple1_conv_mnist(b, batchSize=10000, filterSize=14, filterNum=100):
    imageData = train_set[0].reshape((-1, 1, 28, 28))

    V_val = np.zeros((1, imageData.shape[1], filterSize, filterSize), dtype=np.float32)
    # V_val /= np.sqrt(np.sum(V_val * V_val, axis=(1, 2, 3), keepdims=True))

    B_val = np.ones((filterNum, 1, 1, 1), dtype=np.float32) * b

    W_val, imageMean = batchOpti(imageData, B_val, V_val, batchSize, filterSize, filterNum)

    # return np.transpose(W_val, axes=(0,2,3,1))
    return W_val, imageMean


def makeLogRegFn():
    W = T.fmatrix('W')
    B = T.fvector('B')
    X = T.fmatrix('X')
    Y = T.ivector('Y')
    softmaxOut = T.nnet.softmax(T.dot(X, W) + B)
    loss = -T.mean(T.log(softmaxOut[T.arange(Y.shape[0]), Y]))
    pred = T.argmax(softmaxOut, axis=1)
    errorRate = T.mean(T.neq(pred, Y))
    return theano.function([X, Y, W, B], [loss, errorRate, T.grad(loss, W), T.grad(loss, B)])


def logRegOpti_earlyStop(pipe, lr=[(0.1, 100), (0.01, 100), (0.001, 100)], nClass=10):
    # pipe = lambda x: x
    trainX, trainY = pipe(train_set[0]), train_set[1].astype(np.int32)
    validX, validY = pipe(valid_set[0]), valid_set[1].astype(np.int32)

    nInput = trainX.shape[1]
    W_val = np.zeros((nInput, nClass), dtype=np.float32)
    B_val = np.zeros((nClass,), dtype=np.float32)

    logRegFn = makeLogRegFn()

    print 'Start training'
    bestValidEr = 1.0
    for theta, nItr in lr:
        for i in xrange(nItr):
            loss, er, gW, gB = logRegFn(trainX, trainY, W_val, B_val)
            print 'iter', i, 'loss', loss, 'error rate', er, 'gW mag', np.mean(np.sqrt(np.sum(gW * gW, axis=1))), 'gB mag', np.sqrt(np.sum(gB * gB)),
            W_val -= theta * gW
            B_val -= theta * gB
            validER = logRegFn(validX, validY, W_val, B_val)[1]
            if validER < bestValidEr:
                bestValidEr = validER
            print 'validation er', validER, 'best', bestValidEr



def main():
    # vis.visFilters(simple1_conv_mnist(0.0), 'simple1_conv_C00.png')
    # vis.visFilters(simple1_conv_mnist(-0.1), 'simple1_conv_C01.png')
    # vis.visFilters(simple1_conv_mnist(-0.2), 'simple1_conv_C02.png')
    
    bias = -0.8
    filters, imageMean = simple1_conv_mnist(bias)
    vis.visFilters(filters.transpose((0,2,3,1)), 'simple1_conv.png')

    pipe = lambda S: pipeData(S, imageMean, filters, bias)

    # pipe = None

    logRegOpti_earlyStop(pipe)


if __name__ == '__main__':
    sys.exit(int(main() or 0))
