import sys
import math
import numpy as np
import scipy.misc

def scale01AndSaveImg(imgArr, imgPath):
    assert imgArr.ndim == 2 or imgArr.ndim == 3
    tmp = np.copy(imgArr)
    tmp -= np.min(tmp)
    tmp /= np.max(tmp)
    tmp = np.round(tmp * 255.0).astype('int16')
    scipy.misc.imsave(imgPath, tmp)

def findColNum_almostSqr(k):
    return int(math.ceil(math.sqrt(k)))

def cropsPalette(crops, marginSize=1):
    assert crops.ndim == 4 or crops.ndim == 3
    c = findColNum_almostSqr(crops.shape[0])
    h = crops.shape[1]
    l = crops.shape[2]
    marginVal = np.min(crops)
    if crops.ndim == 3:
        ans = np.ones(((h+marginSize) * c - marginSize, (l+marginSize) * c - marginSize), dtype=np.float32) * marginVal
        for i in xrange(crops.shape[0]):
            x = i // c
            y = i % c
            assert i == x * c + y
            ans[(h+marginSize)*x:(h+marginSize)*x+h, (l+marginSize)*y:(l+marginSize)*y+l] = crops[i, :, :]
        return ans
    else:
        ans = np.ones(((h+marginSize) * c - marginSize, (l+marginSize) * c - marginSize, 3), dtype=np.float32) * marginVal
        for i in xrange(crops.shape[0]):
            x = i // c
            y = i % c
            assert i == x * c + y
            ans[(h+marginSize)*x:(h+marginSize)*x+h, (l+marginSize)*y:(l+marginSize)*y+l, :] = crops[i, :, :, :]
        return ans

def visFilters(filters, imgPath, marginSize=1):
    scale01AndSaveImg(cropsPalette(filters, marginSize=marginSize), imgPath)

# def findHighResp(W_val, pixelMean, imgSavePath, n=10000, k=5):
#     numFilter = W_val.shape[0]
#     filterSize = W_val.shape[2]
#     outputImg = np.zeros(((filterSize + 1) * numFilter, (filterSize + 1) * (k+1), 3), dtype=np.int32)
#     corps = randCorp(n, corpSize=filterSize)

#     for i in xrange(numFilter):
#         tmpFilter = W_val[i, :, :, :]
#         tmpFilter -= np.min(tmpFilter)
#         tmpFilter /= np.max(tmpFilter)
#         tmpFilter = np.round(tmpFilter * 255.0).astype('int32')
#         outputImg[(filterSize + 1) * i:(filterSize + 1) * i + filterSize, 0:filterSize, :] \
#                 = np.transpose(tmpFilter, (1, 2, 0))

#         processed = corps / 255.0 - pixelMean
#         # processed /= np.sqrt(np.sum(processed * processed, axis=(1, 2, 3), keepdims=True))
#         scores = np.sum(processed * W_val[i, :, :, :], axis=(1, 2, 3))
#         indices = scores.argsort()[-k:][::-1]
#         for j in xrange(k):
#             outputImg[(filterSize + 1) * i:(filterSize + 1) * i + filterSize, (filterSize + 1) * (j+1):(filterSize + 1) * (j+1) + filterSize, :] \
#                 = np.transpose(corps[indices[j], :, :, :], (1, 2, 0))
#     scipy.misc.imsave(imgSavePath, outputImg)
