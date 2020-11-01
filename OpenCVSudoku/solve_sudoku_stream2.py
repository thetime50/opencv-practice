import cv2 as cv
import numpy as np
import os
import time
from tensorflow.keras.models import load_model
from solve_sudoku_puzzle import solve_sudoku

CAMERA_RUL = 'rtsp://admin:admin@192.168.1.148:8554/live'  # ip摄像头 #帧率不太对

print(True)

# https://www.cnblogs.com/sirxy/p/12126383.html

import threading


class DummyThread:
    def __init__(self):
        pass
    def start(self):
        self.run()
        

# class Producer(threading.Thread):
class Producer(DummyThread):
    """docstring for Producer"""
    def __init__(self, rtmp_str,state='none',apiPreference =cv.CAP_ANY,init=None,ring=None):
        super(Producer, self).__init__()
        self.rtmp_str = rtmp_str
        # 通过cv中的类获取视频流操作对象cap
        self.cap = cv.VideoCapture(self.rtmp_str,apiPreference)
        # 调用cv方法获取cap的视频帧（帧：每秒多少张图片）
        # fps = self.cap.get(cv.CAP_PROP_FPS)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        print(self.fps)
        # 获取cap视频流的每帧大小
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        self.size = (self.width, self.height)
        print(self.size)
#         定义编码格式mpge-4
        # self.fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
        # 定义视频文件输入对象
#         self.outVideo = cv.VideoWriter('./tempdoc/saveDir1.avi', self.fourcc, self.fps, self.size)

        self.state = state
        self.ring=ring
        if(init):
            init(self)
    def run(self):
        print('in producer')
        try:
            ret, image = self.cap.read()
            while ret:
    #             self.outVideo.write(image)
                # image = cv.pyrDown(image)
                
                wait = int(1000 / int(self.fps))
                wait = max(min(wait,300),20)
                key = cv.waitKey(wait)  # 延迟
                if key & 0xFF ==27:#  == ord('q'):#
                    print('break')
    # #                 self.outVideo.release()
    #                 self.cap.release()
    #                 cv.destroyAllWindows()
                    break
                ret, image = self.cap.read()
                if(self.ring):
                    self.state,image = self.ring(self.state,key,image)
                cv.imshow('dis', image)
        # except Exception as e:
        #     print("error",repr(e))
        finally:
            print('end')
    #         self.outVideo.release()
            self.cap.release()
            cv.destroyAllWindows()

model = None

def init(pd):
    global model
    model = load_model("output/digit_classifier.h5")

def ring(state,key,image):
    global model
    # print(state, key, end="\r")
    if type(image) == type(None):
        return state,image
        
    image = cv.resize(image,(image.shape[1]//2,image.shape[0]//2))
    if(state=="run"):
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
            pass
            # print("****************************",error)
    return state,image

           

if __name__ == '__main__':
    print('run program')
    rtmp_str = CAMERA_RUL
    # producer = Producer(rtmp_str,init=init,ring=ring)  # 开个线程
    producer = Producer(rtmp_str,'run',init=init,ring=ring)  # 开个线程
    producer.start()