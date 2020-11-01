
# Once SudokuNet is successfully trained, we’ll deploy it with our solve_sudoku_puzzle.py script to solve a Sudoku puzzle.
# 部署 应用数独模型

# python solve_sudoku_puzzle.py --model output/digit_classifier.h5 --image Sudoku_puzzle.jpg
# python solve_sudoku_puzzle.py --model output/digit_classifier.h5 --image Sudoku_puzzle.jpg -d 1

# debug
# python pyimagesearch\Sudoku\puzzle.py

# from pyimagesearch.sudoku import extract_digit
from pyimagesearch.sudoku import find_puzzle
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
# from Sudoku import Sudoku
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained digit classifier")
ap.add_argument("-i", "--image", required=True,
	help="path to input Sudoku puzzle image")
ap.add_argument("-d", "--debug", type=int, default=-1,
	help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

print("***", args)

# load the digit classifier from disk
print("[INFO] loading digit classifier...")
model = load_model(args["model"])
# load the input image from disk and resize it
print("[INFO] processing image...")
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)

# find the puzzle in the image and then
(puzzleImage, warped) = find_puzzle(image, debug=args["debug"] > 0)
# initialize our 9x9 Sudoku board
board = np.zeros((9, 9), dtype="int") # 9x9 数独矩阵
# a Sudoku puzzle is a 9x9 grid (81 individual cells), so we can
# infer the location of each cell by dividing the warped image
# into a 9x9 grid

exit(0)

stepX = warped.shape[1] // 9
stepY = warped.shape[0] // 9
# initialize a list to store the (x, y)-coordinates of each cell
# location
cellLocs = []