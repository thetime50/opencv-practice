import cv2 as cv
import numpy as np
import os
import time
from RTSCapture import RTSCapture
from tensorflow.keras.models import load_model
from solve_sudoku_puzzle import solve_sudoku

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
            puzzleImage = solve_result["puzzleImage"]
            puzzleCnt = solve_result["puzzleCnt"]
            print(puzzleCnt)
            
            # image.
            cv.drawContours(image, [puzzleCnt], -1, (0,255,0), 2)
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
            key = cv.waitKey(wait)  # 延迟
            if key & 0xFF ==27:#  == ord('q'):#
                print('break')
                break
            ret, image = rtscap.read_latest_frame() #read_latest_frame() 替代 read()

            state,image = producer(state,key,image)
            if not ret:
                continue
            cv.imshow('dis', image)
    # except Exception as e:
    #     print("error",repr(e))
    finally:
        print('end')
        rtscap.stop_read()
        rtscap.release()
        cv.destroyAllWindows()
