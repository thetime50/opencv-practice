
# Contains two helper utilities for finding the Sudoku puzzle board itself as well as digits therein.
# 搜索数独框和识别数字
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import fourier

def contoursSort(cnt):
    # 四点排序
    cntSort = np.zeros_like(cnt)
    cntSum = np.copy(cnt).sum(2) # 这个加2好像没啥用
    cntSort[0] = cnt[np.argmin(cntSum)] # [0,0]
    cntSort[2] = cnt[np.argmax(cntSum)]  # [w,h]
    diff = np.diff(cnt)
    cntSort[1] =cnt[np.argmin(diff)]  #[w,0] # (y<x)
    cntSort[3] = cnt[np.argmax(diff)] #[0,h] # (y>x)
    return cntSort


def fourPointCorrection(gray,image,thresh,puzzleCnt, debug=False):
    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0,255,0), 2)
        for index,point in enumerate(puzzleCnt):
            cv2.putText(output, str(index),tuple(point[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        cv2.imshow("Puzzle Outline",output)
        cv2.waitKey(0)
    threshRoi = four_point_transform(thresh,puzzleCnt.reshape(4, 2))
    minithresh = cv2.resize(threshRoi,(127,127))
    horizontal_sum = np.sum( minithresh, axis=1)
    vertical_sum = np.sum( minithresh, axis=0)
    horizontal_min = np.min(horizontal_sum)
    vertical_min = np.min(vertical_sum)
    horizontal_sum = horizontal_sum - horizontal_min
    vertical_sum = vertical_sum - vertical_min

    # plt.imshow(minithresh, cmap='gray')
    # plt.show()
    # cv2.waitKey(0)
    def grideCheck(line):
        grideFilter  = [1,0,-1,-0.5,0, 0,0,0,0,0 ,0,-0.5,-1,0]*9 + [1] # 14*9 +1
        match = np.sum( grideFilter*line)
        print("grideCheck:",match)
        return match
    def digitCheck(line):
        # aFilter  = [0,0,1,1,1, 1,1,1,1,1 ,1,1,1,0]*9+[0]
        # l1 = line*aFilter
        # plt.subplot2grid((3,2),(0,0),colspan=2)
        # plt.plot(line)
        l1 = line.copy()
        width = 14
        thre = 15e3
        for i in range(4):
            if(l1[i]<l1[i+1] and l1[i]< thre):
                l1[:i] = l1[i]
        for i in range(9):
            index = (i+1)*width -3
            cnt = 1
            while index+cnt<len(l1) and l1[index+cnt-1]>l1[index+cnt] and cnt<7:
                cnt+=1
            while index+cnt<len(l1) and l1[index+cnt-1]<l1[index+cnt] and cnt<7:
                l1[index+cnt] = l1[index+cnt-1]
                cnt+=1
            while index+cnt<len(l1) and l1[index+cnt-1]>l1[index+cnt] and cnt<7:
                l1[index+cnt] = l1[index+cnt-1]
                cnt+=1

        akArr = np.fft.fft(l1)
        modulus = np.abs(akArr)[:len(l1) // 2]
        modulus[0] =0 
        modulus_9 = modulus[9]
        phase_9 = fourier.getPhase(akArr[9:10])[0]
        only_9 = np.sum(modulus>modulus_9/3)==1
        print(modulus_9,phase_9,only_9)
        if(not only_9):
            return False
        # plt.subplot2grid((3,2),(1,0),colspan=2)
        # plt.plot(l1)
        # plt.subplot2grid((3,2),(2,0),colspan=2)
        # plt.plot(modulus, )
        # plt.show()
        return modulus_9>123000/2 and np.abs(phase_9)<0.12

    if grideCheck(horizontal_sum)>40*1000 and grideCheck(vertical_sum)>40*1000 or \
        digitCheck(horizontal_sum) and digitCheck(vertical_sum):
        warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))# 透视修正后的灰度图
        puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))# 透视修正后的彩图
        return warped,puzzle
    return None

def find_puzzle(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # blurred = cv2.GaussianBlur(gray,(7,7),3) # 高斯模糊
    blurred=gray
    areaThreshold = 28*28*81 * 0.8

    thresh = cv2.adaptiveThreshold(blurred,255, # 自动阈值
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,20)# 高斯权重 阈值处理方式 计算半径 减常量
    thresh = cv2.bitwise_not(thresh)

    # cv2.imshow("thresh1",thresh)


    if debug:
        cv2.imshow("Pussle Thresh", thresh)
        cv2.waitKey(0)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, # 外部轮廓
        cv2.CHAIN_APPROX_SIMPLE) # 保存顶点
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 用面积反序排序(从大到小)

    puzzleCnt = None
    warped=None
    puzzle = None
    
    # print("cnts:",len(cnts))
    for c in cnts: # 从大到小查找4边形轮廓
        peri = cv2.arcLength(c,True) # 轮廓长度
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # 多边形拟合
        approxArea = cv2.contourArea(approx)
        squareArea = (peri/4)**2
        if(approxArea<areaThreshold):
            break
        if len(approx == 4) and approxArea > squareArea*0.6: # 顶点检查 面积检查
            puzzleCnt = contoursSort(approx)
            r = fourPointCorrection(gray,image,thresh,puzzleCnt,debug)
            if(r):
                [warped,puzzle] = r
                break
            else:
                puzzleCnt = None
    
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku pussle outline."
            "Try debugging Your thresholding and contour steps."))
    
    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    # return a 2-tuple of puzzle in both RGB and grayscale
    return {
        "puzzle":puzzle,# 透视修正后的彩图
        "warped":warped,# 透视修正后的灰度图
        "puzzleCnt":puzzleCnt,#题目轮廓 shape == (4,1,2)
    }

def debugShow(title,img,size=(250,250),iswait=True):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title,*size)
    cv2.imshow(title, img)
    iswait and cv2.waitKey(0)

# 分辨过滤出数字单元格 并统一大小
# cell为二维亮度图片进行二值化 输出为笔画图片
# border=[t,b,l,r]
def extract_digit(cell, shape=None, border=[1,1,1,1], debug=False,position=None,check = True):
    cellstr = (str(position) if position else "")
    # thresh = cv2.threshold(cell, 0, 255, # 自动阈值
    #                     cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # 二进制 大津算法 # 返回 阈值,图像
    thresh = cell
    thresh = clear_border(thresh) # 清除边框
    if np.max(cell) - np.min(cell) <255*0.25: # 对比度太低
        return None
    
    if debug:
        debugShow("Cell Thresh" + cellstr, np.concatenate((cell,thresh),axis=1), (500,250))
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts) # 兼容cv版本处理findContours的返回结果
    if len(cnts) == 0:
        return None

    
    c = max(cnts, key=cv2.contourArea) # 获取面积最大的轮廓 todo 包含在中心2/4处
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    (th, tw) = thresh.shape
    x, y, w, h = cv2.boundingRect(c)
    percentFilled = cv2.countNonZero(mask) / float(th* tw) # 轮廓占单元格的比例
    if check and percentFilled < 1/28*0.8: # 面积比线性转换
        return None
    if(check and (x > tw*3/4 or x+w < tw/4 or \
       y > th*3/4 or y+h < th/4)):
        return None
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask) # 用最大的轮廓做一次蒙版 数字笔画是一体的 避免干扰

    if type(shape) != type(None):

        mlen = max(w, h)
        ry = y + (h - mlen)//2
        rx = x + (w - mlen)//2
        rw = rh = mlen
        ry1 = max(ry,0)
        ry2 = min(ry+rh,digit.shape[0]-1)
        rx1 = max(rx,0)
        rx2 = min(rx+rw,digit.shape[1]-1)
        roi =  digit[ry1 : ry2, rx1 : rx2]

        roi = cv2.resize(
            roi,
            (
                shape[0]-border[2]-border[3],
                shape[1]-border[0]-border[1]
            )
        )
        # roiPut = np.zeros(shape,dtype="uint8")
        digit = cv2.copyMakeBorder(
            roi,
            border[0],border[1],border[2],border[3],
            cv2.BORDER_CONSTANT,value=0
        )
        # print("***----",,mask)
        # copyMakeBorder
        # cv2.imshow('roi' + cellstr,digit)
        # cv2.waitKey(0)

    # check to see if we should visualize the masking step
    if debug:
        # cv2.imshow("Digit" + cellstr, digit)
        debugShow("Digit" + cellstr, cv2.resize(digit, (28, 28))) # ,interpolation=cv2.INTER_AREA
    # return the digit to the calling function
    return digit

def get_cell_locs(shape):
    stepX = shape[1] / 9
    stepY = shape[0] / 9

    cellLocs = [] # puzzle cells ROI

    for y in range(0, 9):
        row = [] # row ROIs
        for x in range(0, 9):
            startX = round(x * stepX)
            startY = round(y * stepY)
            endX = round((x + 1) * stepX)
            endY = round((y + 1) * stepY)
            row.append((startX, startY, endX, endY))
        cellLocs.append(row)
    return cellLocs

def analysis_pussle_image(
        warped, # 灰度图像
        puzzleCnt, # 数独轮廓
        model, # 识别模型
        cellCb = None, # 回调
        debug = False,
    ):
    # initialize our 9x9 Sudoku board
    board = np.zeros((9, 9), dtype="int") # 9x9 数独矩阵

    blockSize = max(11,min(warped.shape)//5)
    if(blockSize%2==0): blockSize+=1
    thresh = cv2.adaptiveThreshold(warped,255, # 自动阈值
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,blockSize,50)# 高斯权重 阈值处理方式 计算半径 减常量
    if debug:
        cv2.imshow("thresh",thresh)
        # cv2.waitKey(1)
    stepX = thresh.shape[1] / 9
    stepY = thresh.shape[0] / 9

    cellLocs = [] # puzzle cells ROI

    ocrCells = []
    ocrRoi = []
    for y in range(0, 9):
        row = [] # row ROIs
        for x in range(0, 9):
            startX = round(x * stepX)
            startY = round(y * stepY)
            endX = round((x + 1) * stepX)
            endY = round((y + 1) * stepY)
            row.append((startX, startY, endX, endY))
            
            cell = thresh[startY:endY, startX:endX] # 原图裁切出单元格
            digit = extract_digit(cell,shape = (28, 28), border=[2,2,2,2] , debug=debug,position = (x,y)) # 是字符单元格
            # verify that the digit is not empty
            continue_ =  cellCb and cellCb({
                "cellLocs" : cellLocs, #每个单元格位置
                "xindex":x,
                "yindex":y,
                "digit":digit,
                
                "puzzleCnt" : puzzleCnt, # 数独范围
            })
            if continue_:
                continue
            if digit is not None:
                roi = digit # cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                # roi = np.expand_dims(roi, axis=0) # 从第几个维度的位置插入一个维度 [[item]]
                
                # classify the digit and update the Sudoku board with the
                # prediction
                # presult = model.predict(roi).argmax(axis=1)) # [[0的期望,1的期望...]]
                # presult.argmax(axis=1)) # 在第1维做最值运算
                # pred = model.predict(roi).argmax(axis=1)[0] # ocr 识别数字 一次15ms
                # board[y, x] = pred
                ocrCells.append((x,y))
                ocrRoi.append(roi)
        # add the row to our cell locations
        cellLocs.append(row)
    predictions = model.predict(np.stack(ocrRoi)).argmax(axis=1) # 30ms
    for (x,y),pred in zip(ocrCells,predictions):
        board[y,x] = pred
    return cellLocs, board

if __name__ == "__main__":
    find_puzzle(cv2.imread("./sudoku_puzzle.jpg"),True)
