# 一个清空的视频流处理模板 还没加应用

import cv2 as cv
import numpy as np
import os
import time
import argparse 
import copy
from pyimagesearch.sudoku import find_puzzle, analysis_pussle_image
from tensorflow.keras.models import load_model
from sudoku import Sudoku
from solve_sudoku_puzzle import draw_sudoku_solution_on_src

'''
第一阶段
数独识别提取
结果: 数独轮廓 单元格位置 数独数组
第二阶段
求解
结果：j结果数组
第三阶段
画图 逆变换 显示
'''

print("stream file")


# CAMERA_RUL = 'rtsp://admin:admin@192.168.1.148:8554/live'  # ip摄像头 #帧率不太对
CAMERA_RUL = 'rtsp://admin:admin@192.168.31.60:8554/live'  # ip摄像头 #帧率不太对

import threading
from multiprocessing import Process, Queue,Pool,Manager

class DummyThread:
    def __init__(self):
        pass
    def start(self):
        self.run()
        

# class Producer(threading.Thread):
class Producer(DummyThread):
    """docstring for Producer"""
    def __init__(self, rtmp_str,state='none',apiPreference =cv.CAP_ANY,init=None,ring=None):
        print("Producer __init__")
        super(Producer, self).__init__()
        self.rtmp_str = rtmp_str
        # 通过cv中的类获取视频流操作对象cap
        self.cap = cv.VideoCapture(self.rtmp_str,apiPreference)
        # 调用cv方法获取cap的视频帧（帧：每秒多少张图片）
        # fps = self.cap.get(cv.CAP_PROP_FPS)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        print(f"fps:{self.fps}")
        # 获取cap视频流的每帧大小
        self.width = int(self.cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        
        self.size = (self.width, self.height)
        print(f"size:{self.size}")
#         定义编码格式mpge-4
        # self.fourcc = cv.VideoWriter_fourcc('M', 'P', '4', '2')
        # 定义视频文件输入对象
#         self.outVideo = cv.VideoWriter('./tempdoc/saveDir1.avi', self.fourcc, self.fps, self.size)

        self.state = state
        self.ring=ring
        if(init):
            init(self)
    def run(self):
        print("Producer run")
        # print('in producer')
        try:
            ret, image = self.cap.read()
            while ret:
    #             self.outVideo.write(image)
                # image = cv.pyrDown(image)
                
                wait = int(1000 / int(self.fps))
                wait = max(min(wait,300),20)
                wait = 2
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
            del self.cap
            cv.destroyAllWindows()

class ImgWrap:
    cnt = 0
    def __init__(self,
        img, # 原图
        contour=None, # 轮廓
        correctionImgShape = None, # 修正后的形状
        cellLocs=None, # 单元格位置
        puzzle=None, # 题目
        solution=None, # 结果
    ):
        self.timestamp = time.time()
        self.serial=ImgWrap.cnt
        ImgWrap.cnt+=1

        self.img = img
        self.contour = contour
        # puzzleImage
        self.correctionImgShape = correctionImgShape
        self.cellLocs = cellLocs
        self.puzzle = puzzle
        self.solution = solution

def mainProcess(imgIntQ):
    def init(pd):
        pass
    def ring(state,key,image):
        imgIntQ.put(ImgWrap(image))
        return state,image
    ## start
    rtmp_str = CAMERA_RUL
    producer = Producer(rtmp_str,'load',init=init,ring=ring)
    producer.start()
    print("mpend")


def fcProcess(para):
    (model,imgIntQ,imgPutQ,imgSolveQ) = para

    while True:
        try:
            iw = imgIntQ.get(timeout=300)
            if(iw is None):
                print("fc queue is None")
                # break
            else:
                try:
                    fp_result = find_puzzle(iw.img)
                    
                    puzzleImage = fp_result["puzzle"]# 透视修正后的彩图
                    warped = fp_result["warped"]# 透视修正后的灰度图
                    puzzleCnt = fp_result["puzzleCnt"] # 原图轮廓
                    
                    cellLocs, puzzle = analysis_pussle_image(
                        warped,
                        puzzleCnt,
                        model
                    )

                    iw.contour = puzzleCnt
                    iw.correctionImgShape = puzzleImage.shape
                    iw.cellLocs = cellLocs
                    iw.puzzle = puzzle
                    
                    imgSolveQ.put(iw)
                except Exception as e:
                    pass
                imgPutQ.put(iw)

        except Exception as error:#,Argument
            print("****************************",error)
            # break
        # print('fp',image)

def solveSodokuProcess(imgSolveQ):
    while 1:
        iw = imgSolveQ.get()
        if(iw is None):
            continue
        solution = Sudoku(3,3,iw.puzzle)
        iw.solution = solution.board

def putProcess(imgPutQ,imgSolutionQ):
    buff = []
    buffSize = 5
    before = -1
    while True:
        try:
            iw = imgPutQ.get(timeout=10)
            siw = imgSolutionQ.get(timeout = 0)
            if(not iw is None):
                pass # 插入帧
                if(len(buff)<=0):
                    buff.append(iw)
                else:
                    if(iw.serial == buff[0].serial-1 ):
                        buff.insert(0,iw)
                    elif(iw.serial > buff[0].serial):
                        for i in range(buff):
                            if(iw.serial > buff[i].serial):
                                buff.insert(i+1,iw)
                                break
                                
            shim = buff[0]
            if(shim.serial <= before+1 or len(buff) > buffSize):
                buff.pop()
                before = shim.serial
                # draw
                if(siw and np.all( siw.puzzle == shim.puzzle )):
                    draw_sudoku_solution_on_src(
                        shim.img,
                        shim.contour,
                        shim.correctionImgShape,
                        shim.cellLocs,
                        shim.puzzle,
                        shim.solution
                    )
                cv.imshow('pnp',shim.img)
                cv.waitKey(10)
        except Exception as error:#,Argument
            print("****************************",error)
            break
    cv.destroyAllWindows()

if __name__ == '__main__':
    print('run program')
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained digit classifier")
    ap.add_argument("-i", "--image", required=True,
        help="path to input Sudoku puzzle image")
    args = vars(ap.parse_args())

    model = load_model(args["model"])

    # imgIntQ = Queue()
    # imgPutQ = Queue()
    imgIntQ = Manager().Queue()
    imgPutQ = Manager().Queue()
    imgSolveQ = Manager().Queue()
    imgSolutionQ = Manager().Queue()
    
    mp = Process(name="mainProcess",target=mainProcess, args=(imgIntQ,)) # 获取视频进程
    pp = Process(name="putProcess",target=putProcess, args=(imgPutQ,imgSolutionQ)) # 显示结果进程
    ssp = Process(name="solveSodokuProcess",target=solveSodokuProcess, args=(imgSolveQ,imgSolutionQ)) # 显示结果进程
    mp.start()
    pp.start()
    ssp.start()

    poolcnt = 2
    fpo = Pool(poolcnt) # 处理进程
    fpo.map(fcProcess, [(model,imgIntQ,imgPutQ,imgSolveQ)]*poolcnt)

    mp.join()
    print("mp exit")

    imgIntQ.close()
    imgPutQ.close()
    imgSolveQ.close()
    imgSolutionQ.close()

    # fpo.close()
    # fpo.terminate()
    fpo.join()
    print("fpo exit")

    # ssp.close()
    # ssp.terminate()# 强制退出
    ssp.join()
    print("ssp exit")

    # pp.close()
    # pp.terminate()# 强制退出
    pp.join()
    print("pp exit")

    cv.destroyAllWindows()
    print("all end.....")

    
# close()跟terminate()的区别在于close()会等待池中的worker进程执行结束再关闭pool,而terminate()
'''
Process Pool执行顺序问题
queue.get timeout 问题
'''