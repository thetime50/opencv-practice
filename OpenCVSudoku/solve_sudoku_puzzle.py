
# Once SudokuNet is successfully trained, we’ll deploy it with our solve_sudoku_puzzle.py script to solve a Sudoku puzzle.
# 部署 应用数独模型

# python solve_sudoku_puzzle.py --model output/digit_classifier.h5 --image Sudoku_puzzle.jpg
# python solve_sudoku_puzzle.py --model output/digit_classifier.h5 --image Sudoku_puzzle.jpg -d 1
# python solve_sudoku_puzzle.py --model output/print_digit_classifier.h5 --image Sudoku_puzzle1.jpg
# python solve_sudoku_puzzle.py --model output/mixin_digit_classifier.h5 --image Sudoku_puzzle1.jpg

# debug
# python pyimagesearch\Sudoku\puzzle.py

import h5py # 处理imutils冲突
from pyimagesearch.sudoku import analysis_pussle_image
from pyimagesearch.sudoku import find_puzzle,get_cell_locs
import tensorflow as tf
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2

# tf.debugging.set_log_device_placement(True)

def solve_sudoku(model, image, cellCb=None, debug=False,):

    # find the puzzle in the image and then
    fp_result = find_puzzle(image, debug=debug)
    puzzleImage = fp_result["puzzle"]
    warped = fp_result["warped"]
    puzzleCnt = fp_result["puzzleCnt"] # 原图轮廓

    def cellCb_(args):
        args["puzzleImage"] = puzzleImage # 透视修正后的彩图
        return cellCb(args)
    cellLocs,board = analysis_pussle_image(
        warped,
        puzzleCnt,
        model,
        cellCb and cellCb_,
        debug,
    )
    
    print("puzzleing ...")
    puzzle = Sudoku(3, 3, board=board.tolist())
    solution = puzzle.solve() # 解数独
    return {
        "puzzle" : puzzle, # 求解前的数独
        "solution" : solution, # 求解后的数独
        "cellLocs" : cellLocs, #每个单元格位置
        "puzzleImage" : puzzleImage, # 透视修正后的彩图
        "puzzleCnt" : puzzleCnt, # 数独轮廓
    }

def draw_sudoku_solution(image,cells,puzzleArr,solutionArr = None):
    # loop over the cell locations and board
    if(solutionArr is None):
        solutionArr = [[None for c in range(9)] for r in range(9)]
    for (cellRow, puzzleRow,solutionRow) in zip(cells, puzzleArr,solutionArr):
        # loop over individual cell in the row
        for (box, pDigit,sDigit) in zip(cellRow, puzzleRow,solutionRow):
            # unpack the cell coordinates
            startX, startY, endX, endY = box
            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            showDigit = sDigit
            color = (0, 255, 255)
            if(type(pDigit) != type(None)):
                color = (255,0,0)
                showDigit = pDigit
            # draw the result digit on the Sudoku puzzle image
            if(not showDigit is None):
                cv2.putText(image, str(showDigit), (textX, textY),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def shap2point(shape):
    return np.float32([
        [0,0],
        [shape[1],0],
        [shape[1],shape[0]],
        [0,shape[0]],
    ])

def draw_sudoku_solution_on_src(
    srcImg, # 原图
    contour, # 轮廓
    correctionImgShape, # 透视还原后彩图shape
    puzzleArr, # 题目数组
    solutionArr # 答案数组
):
    tempImg = np.zeros(correctionImgShape,"uint8")
    cellLocs = get_cell_locs(correctionImgShape)
    draw_sudoku_solution(
        tempImg,
        cellLocs,
        puzzleArr,
        solutionArr
    )
    src2des = cv2.getPerspectiveTransform(
        shap2point(correctionImgShape),
        np.float32(contour.reshape(4,2))
    )
    tempImg = cv2.warpPerspective(tempImg, src2des,(srcImg.shape[1],srcImg.shape[0]))
    mask = np.any(tempImg,axis=-1)
    srcImg[mask] = tempImg[mask]
    cv2.drawContours(srcImg,[contour],-1,(100,255,100),2)

if __name__ == "__main__":
    
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained digit classifier")
    ap.add_argument("-i", "--image", required=True,
        help="path to input Sudoku puzzle image")
    ap.add_argument("-d", "--debug", type=int, default=-1,
        help="whether or not we are visualizing each step of the pipeline")
    args = vars(ap.parse_args())

    # 加载模型
    print("[INFO] loading digit classifier...")
    print(args["model"])
    model = load_model(args["model"])

    # 加载图片
    print("[INFO] processing image...")
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=600)

    debug=args["debug"] > 0

    solve_result = solve_sudoku(model,image,debug=debug)

    puzzle =solve_result["puzzle"] 
    solution =solve_result["solution"] 
    cellLocs =solve_result["cellLocs"] 
    puzzleImage =solve_result["puzzleImage"] 
    puzzleCnt = solve_result["puzzleCnt"]

    puzzle.show()
    solution.show_full()

    # draw_sudoku_solution(puzzleImage,cellLocs,puzzle.board,solution.board)

    # # show the output image
    # cv2.imshow("Sudoku Result", puzzleImage)
    # cv2.waitKey(0)

    solutionImg = image.copy()
    draw_sudoku_solution_on_src(
        solutionImg,
        puzzleCnt,
        puzzleImage.shape,
        puzzle.board,
        solution.board
    )
    cv2.imshow("Sudoku Result2", solutionImg)
    cv2.waitKey(0)