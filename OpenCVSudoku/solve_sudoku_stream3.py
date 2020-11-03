import cv2
import numpy as np
import os
import time
from RTSCapture import RTSCapture
from tensorflow.keras.models import load_model
from solve_sudoku_puzzle import solve_sudoku
from solve_sudoku_puzzle import draw_sudoku_solution

CAMERA_RUL = 'rtsp://admin:admin@192.168.1.148:8554/live'  # ip摄像头 #帧率不太对
MODEL_PATH = "output/digit_classifier.h5"

model = None

def producer(state,key,image):
    # if (key & 0xFF ==ord('f')) and  (state == 'none'):
    global model

    if state == 'none':
        try:
            solve_result = solve_sudoku(model,image)
            puzzle = solve_result["puzzle"]
            solution = solve_result["solution"]
            cellLocs = solve_result["cellLocs"]
            puzzleImage = solve_result["puzzleImage"] # 校正后的谜题图片
            puzzleCnt = solve_result["puzzleCnt"] # 原图的谜题框

            puzzle.show()
            solution.show_full()

            cv2.drawContours(image, [puzzleCnt], -1, (0,255,0), 2)

            puzzleDrawings = np.zeros_like(puzzleImage)
            draw_sudoku_solution(puzzleDrawings,cellLocs,solution,puzzle)

            heightImg = puzzleImage.shape[0] # 这个定义不对
            widthImg = puzzleImage.shape[1]
            pts1 = np.float32([ # 原图的谜题框
                [puzzleCnt[0][0]],
                [puzzleCnt[1][0]],
                [puzzleCnt[3][0]],
                [puzzleCnt[2][0]],
            ])
            pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]]) # 校正后的谜题框
            # print(pts1,"\n",pts2)
            # zeros_like
            rectangle2src = cv2.getPerspectiveTransform(pts2, pts1) # INVERSE TRANSFORMATION MATRIX
            imgInvWarp = cv2.warpPerspective(puzzleDrawings, rectangle2src, (image.shape[1], image.shape[0])) # INV IMAGE WARP

            image = cv2.addWeighted(image, 1, imgInvWarp, 1,0)

        except Exception as error:#,Argument
            print("***error***",error)
            pass

    return state,image

if __name__ == '__main__':
    print('run program')
    CAMERA_RUL
    model = load_model(MODEL_PATH)


    rtscap = RTSCapture.create(CAMERA_RUL)
    rtscap.start_read() #启动子线程并改变 read_latest_frame 的指向

    state = 'none'
    try:
        while rtscap.isStarted():

            wait = 3
            key = cv2.waitKey(wait)  # 延迟
            if key & 0xFF ==27:#  == ord('q'):#
                print('break')
                break
            ret, image = rtscap.read_latest_frame() #read_latest_frame() 替代 read()

            state,image = producer(state,key,image)
            if not ret:
                continue
            cv2.imshow('dis', image)
    # except Exception as e:
    #     print("error",repr(e))
    finally:
        print('end')
        rtscap.stop_read()
        rtscap.release()
        cv2.destroyAllWindows()