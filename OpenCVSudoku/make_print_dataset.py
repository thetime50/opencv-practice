import random
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from pyimagesearch.sudoku import extract_digit

matplotlib.interactive(False) # 关闭plt窗口后show()能返回继续

(mtrain_images, mtrain_labels), (mtest_images, mtest_labels) = mnist.load_data()

# # 显示图片
# plt.figure() # 创建或者激活一个图形框
# plt.imshow(mtrain_images[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# print(mtrain_images.shape, mtrain_labels.shape, mtest_images.shape, mtest_labels.shape)
# print(mtrain_images.dtype,mtrain_labels.dtype)
# exit(0)
# # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# # uint8 uint8

# #处理图片数据
# mtrain_images = mtrain_images / 255.0

# mtest_images = mtest_images / 255.0
# # 显示图片和名称
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(mtrain_images[i], cmap=plt.cm.binary)
#     plt.xlabel(mtrain_labels[i])
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

PRINT_SATASET_FILE = DATASET_PATH + 'print_dataset.npy'
PRINT_CLEAR_BORDER_RATE = 0.5
PRINT_STRETCH_RATE = 0.33

## mixin dataset
MIXIN_PRINT_RATE = 0.5
MIXIN_SATASET_FILE = DATASET_PATH + 'mixin_dataset.npy'

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

def generateDataSet(imgsrc, cnt,clearBorderRate = 0):

    imgdata = np.zeros([cnt,*desshap],'uint8')
    labdata = np.zeros([cnt],'uint8')
    print('generate %(cnt)d dataset...'%{'cnt':cnt})
    for i in range(cnt):
        if i%5000 == 0:
            print('generate ',i)
        # img = np.zeros(desshap,'uint8')
        img = imgdata[i]
        randomTrans = [
            [1,1],
            [-1,1],
            [-1,-1],
            [1,-1]
        ]
        # numpy.random.uniform(low=0.0, high=1.0, size=None)

        # srcimg = random

        label = random.randint(0,9)
        labdata[i] = label
        font = random.randint(0,len(datainfo)-1)
        thi = random.randint(0,len(thickness)-1)
        simg = imgsrc[font][label][thi]

        if(clearBorderRate==0 or random.random()>clearBorderRate):
            desPts = shap2point(img.shape)
            srcPts = shap2point(srcshap)
            # 四角随意拉伸偏移量 目标形状为标准(28)
            ranPts = np.float32(
                [
                    [ \
                        random.uniform(0,img.shape[1]*PRINT_STRETCH_RATE),\
                        random.uniform(0,img.shape[0]*PRINT_STRETCH_RATE)\
                    ] for i in range(4)\
                ]
            )

            ranPts = ranPts * randomTrans
            ranPts = ranPts + desPts
            ranPts = np.float32(ranPts)
            src2des = cv2.getPerspectiveTransform(srcPts, ranPts) # srcPts to ranPts
            cv2.warpPerspective(simg, src2des,tuple(desshap),dst = img) # INV IMAGE WARP
        else:
            tempImg = np.zeros(simg.shape, "uint8")
            
            tempPts = shap2point(tempImg.shape)
            srcPts = shap2point(srcshap)
            # 四角随意拉伸偏移量 目标形状为标准
            ranPts = np.float32(
                [
                    [ \
                        random.uniform(0,tempImg.shape[1]*PRINT_STRETCH_RATE),\
                        random.uniform(0,tempImg.shape[0]*PRINT_STRETCH_RATE)\
                    ] for i in range(4)\
                ]
            )

            ranPts = ranPts * randomTrans
            ranPts = ranPts + tempPts
            ranPts = np.float32(ranPts)
            src2temp = cv2.getPerspectiveTransform(srcPts, ranPts) # srcPts to ranPts
            cv2.warpPerspective(simg, src2temp,tuple(tempImg.shape),dst = tempImg) # INV IMAGE WARP
            tempImg = extract_digit(255-tempImg,shape = img.shape, border=[2,2,2,2])
            np.copyto(img,tempImg)
        
        # plt.figure()
        # plt.imshow(imgdata[i], cmap=plt.cm.binary)
        # plt.xlabel(labdata[i])
        # plt.show()

    return (imgdata,labdata)

def saveImgset(path,prefix,imgdata,labdata):
    for idx,(img,lab) in enumerate(zip(imgdata,labdata)):
        pathfile = '%(path)s%(prefix)s_sn%(idx)06d_%(lab)d.bmp'%{
                'path':path,
                'prefix':prefix,
                'idx':idx,
                'lab':lab
            }
        # print(pathfile)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(pathfile,img) #需要新建文件夹 不然也不会报错


# testDataset = generateDataSet(imgsrc,90)
# saveImgset(
#     './'+DATASET_IMG_PATH,
#     'test',
#     testDataset[0],
#     testDataset[1]
# )

# 处理np.array((),dtype=object)自动广播报错问题
def objectArray(*args):
    res = np.zeros( len(args),object)
    for i,item in enumerate(args):
        res[i] = item
    return res

# 生成数据集

(trainData, trainLabels) = generateDataSet(imgsrc,60000,PRINT_CLEAR_BORDER_RATE)
(testData, testLabels) = generateDataSet(imgsrc,10000,PRINT_CLEAR_BORDER_RATE)

print('save')

np.save( # 会覆盖旧文件
    PRINT_SATASET_FILE,
    (
        objectArray(trainData, trainLabels),
        objectArray(testData, testLabels)
    )
)

def testShowData(title,imgs,labs):
    imgs = imgs[:25] / 255.0

    # 显示图片和名称
    plt.figure(figsize=(10,10)).canvas.manager.set_window_title(title)
    # plt.title(title)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(imgs[i], cmap=plt.cm.binary)
        plt.xlabel(labs[i])
    plt.show() # block=False

testShowData(
    "print dataset",
    trainData,
    trainLabels
)

# testDataset = np.load(PRINT_SATASET_FILE, allow_pickle=True)
# # Object arrays cannot be loaded when allow_pickle=False
# print(testDataset[0][0].shape)

# 生成混合数据集

def mixinDataset(
    rate, # data 1 rate
    d1Data, d1Labels,
    d2Data, d2Labels,
):
    if d1Data.shape != d2Data.shape or \
        d1Labels.shape != d2Labels.shape :
        print(
            'd1Data.shape',d1Data.shape, 'd2Data.shape', d2Data.shape,
            '\nd1Labels.shape',d1Labels.shape, 'd2Labels.shape', d2Labels.shape 
        )
        raise Exception("dataset shape error")
    if len(d1Data) != len(d1Labels):
        raise Exception("dataset length error")

    data = []
    labels = []
    for i in range(len(d1Data)):
        r = random.random()
        if r<rate:
            data.append(d1Data[i])
            labels.append(d1Labels[i])
        else:
            data.append(d2Data[i])
            labels.append(d2Labels[i])
    data = np.array(data,dtype=d1Data.dtype)
    labels = np.array(labels,dtype=d1Labels.dtype)
    return ( data, labels)



(mixinTrainData, mixinTrainLabel) = mixinDataset(
    MIXIN_PRINT_RATE,
    trainData,trainLabels,
    mtrain_images,mtrain_labels
)


(mixinTeseData, mixinTeseLabel) = mixinDataset(
    MIXIN_PRINT_RATE,
    testData,testLabels,
    mtest_images,mtest_labels
)


np.save( # 会覆盖旧文件
    MIXIN_SATASET_FILE,
    (
        objectArray(mixinTrainData, mixinTrainLabel),
        objectArray(mixinTeseData, mixinTeseLabel)
    )
)

testDataset = np.load(MIXIN_SATASET_FILE, allow_pickle=True)
testShowData(
    "mixin dataset",
    testDataset[0][0],
    testDataset[0][1]
)