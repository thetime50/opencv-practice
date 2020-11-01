
# Contains two helper utilities for finding the Sudoku puzzle board itself as well as digits therein.
# 搜索数独框和识别数字

from imutils.perspective import four_point_transform
# from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

def find_puzzle(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray,(7,7),3) # 高斯模糊

    thresh = cv2.adaptiveThreshold(blurred,255, # 自动阈值
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    thresh = cv2.bitwise_not(thresh)

    if debug:
        cv2.imshow("Pussle Thresh", thresh)
        cv2.waitKey(0)
    
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True) # 用面积反序排序

    puzzleCnt = None
    for c in cnts:
        peri = cv2.arcLength(c,True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
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
    
    return None,None



if __name__ == "__main__":
    find_puzzle(cv2.imread("./sudoku_puzzle.jpg"),True)
