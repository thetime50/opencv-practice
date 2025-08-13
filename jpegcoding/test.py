import jpeg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

image = Image.open("../img/color.webp")
ycbcr = image.convert('YCbCr')
# -128?

img = cv2.imread("../img/color.webp")
side = 8
fill = 16
# fill
height = img.shape[0]
width = img.shape[1]
fy = 0
fx = 0
ymod = height%fill
xmod = width%fill
if(ymod):
    fy = fill-ymod
if(xmod):
    fx = fill - xmod
fimg = np.pad(img,((0,fy),(0,fx),(0,0)),'constant')
fimg[:,width:] = fimg[:,width-1:width]
fimg[height:,:] = fimg[height-1:height,:]


yuvimg = cv2.cvtColor(fimg,cv2.COLOR_BGR2YCrCb)

imgy = yuvimg[:,:,0]
imgcr = yuvimg[:,:,1]
imgcb = yuvimg[:,:,2]

# plt.subplot2grid((1,3),(0,0))
# plt.imshow(imgy)
# plt.subplot2grid((1,3),(0,1))
# plt.imshow(imgcr)
# plt.subplot2grid((1,3),(0,2))
# plt.imshow(imgcb)
# plt.show()

# YUV 420
cryidx = np.array(range(0,fimg.shape[0],2))
crxidx = np.array(range(0,fimg.shape[1],2))
imgcr = imgcr[cryidx,:][:,crxidx]
cbyidx = cryidx+1
cbxidx = crxidx+1
imgcb = imgcb[cbyidx,:][:,crxidx]

dcty = np.array(jpeg.dct2dBlock(imgy,8)) #,fill
dctcr = np.array(jpeg.dct2dBlock(imgcr,8)) #,fill/2
dctcb= np.array(jpeg.dct2dBlock(imgcb,8)) #,fill/2

def dctShow(roi):
    roiyuv = cv2.cvtColor(roi,cv2.COLOR_BGR2YCrCb)
    imgy = roiyuv[:,:,0].astype('float32')
    imgcr = roiyuv[:,:,1].astype('float32')
    imgcb = roiyuv[:,:,2].astype('float32')
    [[myroiy]] = jpeg.dct2dBlock(imgy,8) #,fill
    [[myroicb]] = jpeg.dct2dBlock(imgcr,8) #,fill/2
    [[myroicr]]= jpeg.dct2dBlock(imgcb,8) #,fill/2

    
    roiy = cv2.dct(imgy)
    roicb = cv2.dct(imgcr)
    roicr = cv2.dct(imgcb)

    # myroiy[0][0] = myroicb[0][0] = myroicr[0][0] = 0
    # roiy[0][0] = roicb[0][0] = roicr[0][0] = 0

    plt.subplot(331)
    plt.imshow(roi)
    plt.title("origin")
    plt.subplot(334)
    plt.imshow(myroiy,'gray')
    plt.title("my y dct")
    plt.subplot(335)
    plt.imshow(myroicb,'gray')
    plt.title("my cb dct")
    plt.subplot(336)
    plt.imshow(myroicr,'gray')
    plt.title("my cr dct")
    plt.subplot(337)
    plt.imshow(roiy,'gray')
    plt.title("y dct")
    plt.subplot(338)
    plt.imshow(roicb,'gray')
    plt.title("cb dct")
    plt.subplot(339)
    plt.imshow(roicr,'gray')
    plt.title("cr dct")
    plt.show()

roiy = 256
roix = 128
dctShow(img[roiy:roiy+8,roix:roix+8])

def concatBlock(barr):
    bh = barr.shape[0]
    bw = barr.shape[1]
    yside = barr.shape[2]
    xside = barr.shape[3]
    res = np.empty([
        bh*yside,
        bw*xside],
        dtype="float64")
    for by,row in enumerate( barr):
        y = by*yside
        for bx,block in enumerate(row):
            x = bx*xside
            res[y:y+yside,x:x+xside] = block
    return res
plt.subplot(141)
plt.imshow(fimg)
plt.title("origin")
plt.subplot(142)
plt.imshow(concatBlock(dcty),'gray')
plt.title("y")
plt.subplot(143)
plt.imshow(concatBlock(dctcb),'gray')
plt.title("cb")
plt.subplot(144)
plt.imshow(concatBlock(dctcr),'gray')
plt.title("cr")
plt.show()

# 量化
# 标准亮度量化表
Qy = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
               [12, 12, 14, 19, 26, 58, 60, 55],
               [14, 13, 16, 24, 40, 57, 69, 56],
               [14, 17, 22, 29, 51, 87, 80, 62],
               [18, 22, 37, 56, 68, 109, 103, 77],
               [24, 35, 55, 64, 81, 104, 113, 92],
               [49, 64, 78, 87, 103, 121, 120, 101],
               [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.uint8)

Qc = np.array([[17, 18, 24, 47, 99, 99, 99, 99,],
               [18, 21, 26, 66, 99, 99, 99, 99,],
               [24, 26, 56, 99, 99, 99, 99, 99,],
               [47, 66, 99, 99, 99, 99, 99, 99,],
               [99, 99, 99, 99, 99, 99, 99, 99,],
               [99, 99, 99, 99, 99, 99, 99, 99,],
               [99, 99, 99, 99, 99, 99, 99, 99,],
               [99, 99, 99, 99, 99, 99, 99, 99,]])
quality_scale = 50
# 根据压缩质量重新计算量化表
if quality_scale <= 0:
    quality_scale = 1
elif quality_scale >= 100:
    quality_scale = 99
Qy = (Qy * quality_scale+50)/100
Qy[Qy<=0] = 1
Qy[Qy>255] = 255


dcty = np.round(dcty/Qy).reshape([dcty.shape[0],dcty.shape[1],8*8])
dctcr = np.round(dctcr/Qc).reshape([dctcr.shape[0],dctcr.shape[1],8*8])
dctcb = np.round(dctcb/Qc).reshape([dctcb.shape[0],dctcb.shape[1],8*8])

# zig
ZigZag = [
        0, 1, 5, 6, 14, 15, 27, 28,
        2, 4, 7, 13, 16, 26, 29, 42,
        3, 8, 12, 17, 25, 30, 41, 43,
        9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63]

dcty[:,:,:] = dcty[:,:,ZigZag]
dctcr[:,:,:] = dctcr[:,:,ZigZag]
dctcb[:,:,:] = dctcb[:,:,ZigZag]


# rle
# huffma

pass