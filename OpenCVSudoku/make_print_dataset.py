import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# # 显示图片
# plt.figure() # 创建或者激活一个图形框
# plt.imshow(train_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
# print(train_images.dtype,train_labels.dtype)
# exit(0)
# # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# # uint8 uint8

# #处理图片数据
# train_images = train_images / 255.0

# test_images = test_images / 255.0
# # 显示图片和名称
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(train_labels[train_labels[i]])
# plt.show()

# https://docs.opencv.org/master/d6/d6e/group__imgproc__draw.html#ga5126f47f883d730f633d74f07456c576
# putText()
# Parameters
#     img	Image.
#     text	Text string to be drawn.
#     org	Bottom-left corner of the text string in the image. #位置
#     fontFace	Font type, see HersheyFonts. # 字体
#     fontScale	Font scale factor that is multiplied by the font-specific base size. # 缩放
#     color	Text color.
#     thickness	Thickness of the lines used to draw a text. # 线宽
#     lineType	Line type. See LineTypes # 线形
#     bottomLeftOrigin	When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.

DATASET_PATH = 'dataset/'
DATASET_IMG_PATH = DATASET_PATH + 'img/'

datainfo=[
    {'text':'FONT_HERSHEY_SIMPLEX',         'font':cv2.FONT_HERSHEY_SIMPLEX,'count':10000},
    # {'text':'FONT_HERSHEY_PLAIN',           'font':cv2.FONT_HERSHEY_PLAIN ,'count':10000},
    {'text':'FONT_HERSHEY_DUPLEX',          'font':cv2.FONT_HERSHEY_DUPLEX ,'count':10000},
    {'text':'FONT_HERSHEY_COMPLEX',         'font':cv2.FONT_HERSHEY_COMPLEX ,'count':10000},
    {'text':'FONT_HERSHEY_TRIPLEX',         'font':cv2.FONT_HERSHEY_TRIPLEX ,'count':10000},
    # {'text':'FONT_HERSHEY_COMPLEX_SMALL',   'font':cv2.FONT_HERSHEY_COMPLEX_SMALL ,'count':10000},
    {'text':'FONT_HERSHEY_SCRIPT_SIMPLEX',  'font':cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,'count':10000},
    {'text':'FONT_HERSHEY_SCRIPT_COMPLEX',  'font':cv2.FONT_HERSHEY_SCRIPT_COMPLEX ,'count':10000},
    {'text':'FONT_ITALIC',                  'font':cv2.FONT_ITALIC ,'count':10000},
]

thickness=[2,4,6,8]

dataset = []

srcshap = [28*2,28*2]
desshap = [28,28]
rendomshape = np.uint16(desshap)//2

imgsrc = []
for index,item in enumerate(datainfo):
    row = []
    print(index,item)
    for num in range(10):
        thi = []
        for t in thickness:
            im = np.zeros(srcshap,np.uint8)
            cv2.putText(
                im,
                str(num),
                (8,48), #(x,y)
                item['font'],
                2,255,t,
                # bottomLeftOrigin=True
            )
            thi.append(im)
        row.append(thi)
    imgsrc.append(row)

imgsrc = np.asarray(imgsrc,dtype = 'uint8')

# print(imgsrc[0][0])
def showCell(win,data):
    plt.figure(win,figsize=(len(data),10))
    for rindax,row in enumerate(data):
        for cindex,cell in enumerate(row):

            plt.subplot(len(data),10,rindax*10 + cindex + 1)
            # plt.xticks([])
            # plt.yticks([])
            plt.grid(False)
            plt.imshow(cell,vmin=0, vmax=255) # cmap=plt.cm.binary)
            plt.xlabel(datainfo[rindax]['text']+'-'+str(cindex))
    plt.show()

# for index,item in enumerate(thickness):
#     showdata = imgsrc[:,:,index]
#     showCell(index,showdata)

# cv2.imshow('cv0',imgsrc[0][5][0])
# cv2.imshow('cv3',imgsrc[0][5][3])
# cv2.waitKey(0)



# save ((trainData, trainLabels),(testData, testLabels))
# DATASET_PATH
# DATASET_IMG_PATH
# 0---1
# |   |
# 3---2
def shap2point(shape):
    return np.float32([
        [0,0],
        [shape[1],0],
        [shape[1],shape[0]],
        [0,shape[0]],
    ])

def generateDataSet(imgsrc, cnt):
    imgdata = np.zeros([cnt,*desshap],'uint8')
    labdata = np.zeros([cnt],'uint8')
    for i in range(cnt):
        # img = np.zeros(desshap,'uint8')
        img = imgdata[i]
        desPts = shap2point(img.shape)
        srcPts = shap2point(srcshap)
        randomTrans = [
            [1,1],
            [-1,1],
            [-1,-1],
            [1,-1]
        ]
        # numpy.random.uniform(low=0.0, high=1.0, size=None)
        ranPts = np.float32(
            [[random.uniform(0,rendomshape[0]),random.uniform(0,rendomshape[1])] for i in range(4)]
        )

        ranPts = ranPts * randomTrans
        ranPts = ranPts + desPts

        # srcimg = random

        label = random.randint(0,9)
        labdata[i] = label
        font = random.randint(0,len(datainfo)-1)
        simg = imgsrc[font][label]

        ranPts = np.float32(ranPts)
        src2des = cv2.getPerspectiveTransform(srcPts, ranPts) # srcPts to ranPts
        img = cv2.warpPerspective(simg, src2des,tuple(desshap),img) # INV IMAGE WARP

    return (imgdata,labdata)

def saveImgset(path,prefix,imgdata,labdata):
    for idx,(img,lab) in enumerate(zip(imgdata,labdata)):
        pathfile = '%(path)s%(prefix)s_sn%(idx)06d_%(lab)d.bmp'%{
                'path':path,
                'prefix':prefix,
                'idx':idx,
                'lab':lab
            }
        print(pathfile)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(pathfile,img)


testDataset = generateDataSet(imgsrc,30)

saveImgset(
    './'+DATASET_IMG_PATH,
    'test',
    testDataset[0],
    testDataset[1]
)