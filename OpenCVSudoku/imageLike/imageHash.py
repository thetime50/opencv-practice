import cv2
import numpy as np

def test_to_gray(img):
    if(len(img.shape) == 3):
        if(img.shape[2] ==3):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        elif(img.shape[2] == 1):
            img.reshape(img.shape[:-1])
        else:
            raise Exception(f"img.shape 不支持{img.shape}")
        return img
    elif(len(img.shape) == 2):
        return img
    else:
        raise Exception(f"img.shape 不支持{img.shape}")


def image_d_hash(img,shape = (64,65)):
    img = test_to_gray(img)
    if( np.any( np.array( img.shape) != shape)):
        img = cv2.resize(img, shape,interpolation=cv2.INTER_AREA)
    diff = img[:,0:-1] > img[:,1:]
    return diff


def image_p_hash(img,normalShape = (32,32), putShape = (8,8),ignoreDC = False):
    img = test_to_gray(img)
    img = cv2.resize(img,normalShape,interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(img))
    if(ignoreDC): dct[0,0] = 0
    dct_roi = dct[0:putShape[0],0:putShape[1]]
    vareage = np.mean(dct_roi)
    hash = dct_roi > vareage
    return hash

def hash_diff(hash1,hash2):
    return np.sum( hash1!=hash2)

