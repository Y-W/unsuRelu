from glob import glob
from random import shuffle, randrange
import numpy as np
import scipy
import scipy.misc
imagePathPattern = '../../data/*.JPEG'

def getImage256Tensor(k):
    imageLst = glob(imagePathPattern)
    shuffle(imageLst)
    imageLst = imageLst[:k]

    imageData256 = np.zeros((len(imageLst), 3, 256, 256), dtype=np.float32)
    for i, imagePath in enumerate(imageLst):
        im = scipy.misc.imread(imagePath, mode='RGB')
        r, c = im.shape[:2]
        ratio = 256.0 / min(r, c)
        targetShape = (int(round(ratio * r)), int(round(ratio * c)))
        im = scipy.misc.imresize(im, targetShape)
        r_lowBound = int(im.shape[0]*0.5) - 128
        c_lowBound = int(im.shape[1]*0.5) - 128
        im = im[r_lowBound:r_lowBound+256, c_lowBound:c_lowBound+256, :]
        imageData256[i, :, :, :] = np.transpose(im, (2, 0, 1)) / 255.0

    return imageData256

def randColorCorp(n, corpSize):
    result = np.zeros((n, corpSize, corpSize, 3), dtype=np.float32)
    imageLst = glob(imagePathPattern)
    shuffle(imageLst)
    randSampNum = {}
    for i in xrange(n):
        rImg = imageLst[randrange(len(imageLst))]
        if rImg in randSampNum:
            randSampNum[rImg] += 1
        else:
            randSampNum[rImg] = 1
    k = 0
    for imgPath, cnt in randSampNum.iteritems():
        im = scipy.misc.imread(imgPath, mode='RGB')
        r, c = im.shape[:2]
        for j in xrange(cnt):
            x, y = randrange(r - corpSize + 1), randrange(c - corpSize + 1)
            result[k, :, :, :] = im[x:x+corpSize, y:y+corpSize, :] / 255.0
            k += 1
    assert k == n
    return result
