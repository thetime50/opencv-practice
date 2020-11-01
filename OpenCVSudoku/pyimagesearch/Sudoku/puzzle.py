
# Contains two helper utilities for finding the Sudoku puzzle board itself as well as digits therein.
# 搜索数独框和识别数字

from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray,(7,7),3) # 高斯模糊

    thresh = cv2.adaptiveThreshold(blurred,255, # 自动阈值
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)# 高斯权重 阈值处理方式 计算半径 减常量
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow("Pussle Thresh", thresh)
        cv2.waitKey(0)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, # 外部轮廓
        cv2.CHAIN_APPROX_SIMPLE) # 保存顶点
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 用面积反序排序

    puzzleCnt = None
    for c in cnts:
        peri = cv2.arcLength(c,True) # 轮廓长度
        approx = cv2.approxPolyDP(c, 0.02 * peri, True) # 多边形拟合
        if len(approx == 4):
            puzzleCnt = approx
            break
    
    if puzzleCnt is None:
        raise Exception(("Could not find Sudoku pussle outline."
            "Try debugging Your thresholding and contour steps."))

    if debug:
        output = image.copy()
        cv2.drawContours(output, [puzzleCnt], -1, (0,255,0), 2)
        cv2.imshow("Puzzle Outline",output)
        cv2.waitKey(0)

    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the puzzle
    puzzle = four_point_transform(image, puzzleCnt.reshape(4, 2))# 透视修正后和灰度图
    warped = four_point_transform(gray, puzzleCnt.reshape(4, 2))# 透视修正后的彩图
    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("Puzzle Transform", puzzle)
        cv2.waitKey(0)
    # return a 2-tuple of puzzle in both RGB and grayscale
    return (puzzle, warped)

# 提取单元格内数字
def extract_digit(cell, debug=False):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255, # 自动阈值
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1] # 二进制 大津算法 # 返回 阈值,图像
    thresh = clear_border(thresh) # 清除边框
    # check to see if we are visualizing the cell thresholding step
    if debug:
        cv2.imshow("Cell Thresh", thresh)
        cv2.waitKey(0)
    
    # find contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    # if no contours were found than this is an empty cell
    if len(cnts) == 0:
        return None
    # otherwise, find the largest contour in the cell and create a
    # mask for the contour
    c = max(cnts, key=cv2.contourArea)
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # compute the percentage of masked pixels relative to the total
    # area of the image
    (h, w) = thresh.shape
    percentFilled = cv2.countNonZero(mask) / float(w * h)
    # if less than 3% of the mask is filled then we are looking at
    # noise and can safely ignore the contour
    if percentFilled < 0.03:
        return None
    # apply the mask to the thresholded cell
    digit = cv2.bitwise_and(thresh, thresh, mask=mask)
    # check to see if we should visualize the masking step
    if debug:
        cv2.imshow("Digit", digit)
        cv2.waitKey(0)
    # return the digit to the calling function
    return digit

if __name__ == "__main__":
    find_puzzle(cv2.imread("./sudoku_puzzle.jpg"),True)
