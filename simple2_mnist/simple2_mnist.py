import sys, os
import numpy as np
import theano, theano.tensor as T

utilsPath = '../utils'
sys.path.append(os.path.abspath(utilsPath))
import vis

import pickle, gzip
with gzip.open('../../mnist/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)


def makeFn():
    X = T.fmatrix('X')
    W = T.fmatrix('W')
    B = T.fvector('B')

    acti = T.nnet.relu(T.dot(X, W) + B)
    recon = T.dot(acti, T.transpose(W))

    diff = X - recon
    loss = T.mean(diff * diff)

    return theano.function([X, W, B], [loss, T.grad(loss, W), T.grad(loss, B)])


def opti_sgd(schema, outputVisImg, k=100):
    nIn = train_set[0].shape[1]
    W_val = np.random.normal(size=(nIn, k)).astype(np.float32)
    W_val /= np.sqrt(np.sum(W_val * W_val, axis=0, keepdims=True))
    B_val = np.zeros((k,), dtype=np.float32)

    fn = makeFn()
    for lr, nItr in schema:
        for i in xrange(nItr):
            loss, gW, gB = fn(train_set[0], W_val, B_val)
            print 'Itr', i, 'LR', lr, 'Loss', loss
            print np.min(B_val), np.mean(B_val), np.max(B_val)
            W_val -= lr * gW
            B_val -= lr * gB
    vis.visFilters(W_val.T.reshape((k, 28, 28)), outputVisImg)


def main():
    opti_sgd([(1e0, 1000), (1e-1, 1000)], 'simple2_filters.png')

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))