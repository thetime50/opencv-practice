
# Once SudokuNet is successfully trained, we’ll deploy it with our solve_sudoku_puzzle.py script to solve a Sudoku puzzle.
# 部署 应用数独模型

# python solve_sudoku_puzzle.py --model output/digit_classifier.h5 --image Sudoku_puzzle.jpg
# python solve_sudoku_puzzle.py --model output/digit_classifier.h5 --image Sudoku_puzzle.jpg -d 1

# debug
# python pyimagesearch\Sudoku\puzzle.py

from pyimagesearch.sudoku import extract_digit
from pyimagesearch.sudoku import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2


def solve_sudoku(model, image, debug=False):

    # find the puzzle in the image and then
    fp_result = find_puzzle(image, debug=debug)
    puzzleImage = fp_result["puzzle"]
    warped = fp_result["warped"]
    puzzleCnt = fp_result["puzzleCnt"]
    # initialize our 9x9 Sudoku board
    board = np.zeros((9, 9), dtype="int") # 9x9 数独矩阵

    stepX = warped.shape[1] // 9
    stepY = warped.shape[0] // 9

    cellLocs = [] # puzzle cells ROI

    for y in range(0, 9):
        row = [] # row ROIs
        for x in range(0, 9):
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY
            row.append((startX, startY, endX, endY))
            
            cell = warped[startY:endY, startX:endX] # 原图裁切出单元格
            digit = extract_digit(cell, debug=debug) # 是字符单元格
            # verify that the digit is not empty
            if digit is not None:
                roi = cv2.resize(digit, (28, 28))
                roi = roi.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0) # 从第几个维度的位置插入一个维度 [[item]]
                # classify the digit and update the Sudoku board with the
                # prediction
                # presult = model.predict(roi).argmax(axis=1)) # [[0的期望,1的期望...]]
                # presult.argmax(axis=1)) # 在第1维做最值运算
                pred = model.predict(roi).argmax(axis=1)[0] # ocr 识别数字
                board[y, x] = pred
        # add the row to our cell locations
        cellLocs.append(row)

    puzzle = Sudoku(3, 3, board=board.tolist())
    solution = puzzle.solve() # 解数独
    return {
        "puzzle" : puzzle,
        "solution" : solution,
        "cellLocs" : cellLocs,
        "puzzleImage" : puzzleImage,
        "puzzleCnt" : puzzleCnt,
    }

def draw_sudoku_solution(image,cells,solution,puzzle = None):
    # loop over the cell locations and board
    for (cellRow, boardRow) in zip(cells, solution.board):
        # loop over individual cell in the row
        for (box, digit) in zip(cellRow, boardRow):
            # unpack the cell coordinates
            startX, startY, endX, endY = box
            # compute the coordinates of where the digit will be drawn
            # on the output puzzle image
            textX = int((endX - startX) * 0.33)
            textY = int((endY - startY) * -0.2)
            textX += startX
            textY += endY
            # draw the result digit on the Sudoku puzzle image
            cv2.putText(image, str(digit), (textX, textY),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

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
    model = load_model(args["model"])

    # 加载图片
    print("[INFO] processing image...")
    image = cv2.imread(args["image"])
    image = imutils.resize(image, width=600)

    solve_result = solve_sudoku(model,image,debug=args["debug"] > 0)

    puzzle =solve_result["puzzle"] 
    solution =solve_result["solution"] 
    cellLocs =solve_result["cellLocs"] 
    puzzleImage =solve_result["puzzleImage"] 

    puzzle.show()
    solution.show_full()

    draw_sudoku_solution(puzzleImage,cellLocs,solution,puzzle)

    # show the output image
    cv2.imshow("Sudoku Result", puzzleImage)
    cv2.waitKey(0)