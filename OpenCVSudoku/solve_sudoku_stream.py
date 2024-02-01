# 一个清空的视频流处理模板 还没加应用

import cv2 as cv
import numpy as np
# import os
import time
import argparse 
import copy
from pyimagesearch.sudoku import find_puzzle, analysis_pussle_image
from tensorflow.keras.models import load_model
from sudoku import Sudoku
from solve_sudoku_puzzle import draw_sudoku_solution_on_src
from imageLike.imageHash import image_d_hash,hash_diff
import sys
import traceback
import inspect

# python solve_sudoku_stream.py --model output/mixin_digit_classifier.h5

'''
第一阶段
数独识别提取
结果: 数独轮廓 单元格位置 数独数组
第二阶段
求解
结果：j结果数组
第三阶段
画图 逆变换 显示

队列阻塞检查
'''

print("stream file")


# CAMERA_RUL = 'rtsp://admin:admin@192.168.1.148:8554/live'  # ip摄像头 #帧率不太对
CAMERA_RUL = 'rtsp://admin:admin@192.168.31.60:8554/live'  # ip摄像头 #帧率不太对

MAX_QUEUE_SIZE = 6

# import threading
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
                # cv.imshow('dis', image)
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
    sudokuCnt = 0
    def __init__(self,
        img, # 原图
        sudokuSerial=None, # 求解序号
        contour=None, # 轮廓
        warped=None, # 透视修正后的灰度图
        correctionImgShape = None, # 修正后的形状
        cellLocs=None, # 单元格位置
        puzzle=None, # 题目
        solution=None, # 结果
    ):
        self.timestamp = time.time()
        self.serial=ImgWrap.cnt
        self.sudokuSerial = sudokuSerial
        ImgWrap.cnt+=1

        self.img = img
        self.contour = contour
        self.warped = warped
        # puzzleImage
        self.correctionImgShape = correctionImgShape
        self.cellLocs = cellLocs
        self.puzzle = puzzle
        self.solution = solution
    def setNewSudoku(self):
        ImgWrap.sudokuCnt += 1
        self.sudokuSerial = ImgWrap.sudokuCnt
    def setLikeSudoku(self):
        self.sudokuSerial = ImgWrap.sudokuCnt

def mainProcess(imgIntQ):
    def init(pd):
        pass
    def ring(state,key,image):
        if(imgIntQ.qsize()<MAX_QUEUE_SIZE):
            imgIntQ.put(ImgWrap(image))
        return state,image
    ## start
    rtmp_str = CAMERA_RUL
    producer = Producer(rtmp_str,'load',init=init,ring=ring)
    producer.start()
    print("mpend")

def imgLike(hash1,hash2):
    d = hash_diff(hash1,hash2)/(hash2.shape[0]*hash2.shape[1])
    return d<0.3
    #位移 缩放 旋转 畸变 亮度 时间
def contourLike(contour1,contour2):
    # vectors = []
    c1 = contour1
    c2 = contour2
    if(len(c1.shape) == 3):
        c1 = np.reshape(c1,(4,2))
    if(len(c2.shape) == 3):
        c2 = np.reshape(c2,(4,2))
    length = 0
    for p1,p2 in zip(c1,c2):
        vi = np.array(p2) - np.array(p1)
        l = np.linalg.norm(vi)
        length += l
    rate = length / cv.arcLength(contour1,closed=True)
    return rate<0.1


# 实时批量处理 有sudoku判断
def fcProcess(para):
    (imgIntQ,imgPutQ,imgSolveM) = para
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

                    iw.contour = puzzleCnt
                    iw.warped = warped
                    iw.correctionImgShape = puzzleImage.shape
                    imgHash = image_d_hash(warped)

                    iwMap = imgSolveM.value
                    ilike = None
                    if(iwMap is not None):
                        oldImgHash = iwMap['oldInfo']['imgHash']
                        oldContour = iwMap['oldInfo']['contour']  
                        clike = contourLike(oldContour,puzzleCnt)
                        ilike = clike and imgLike(oldImgHash,imgHash)
                    # 加一个求解序号 相同画面用同一个序号
                    if( ilike):

                        iw.setLikeSudoku()
                        imgSolveM.value = {
                            "iw":iw,
                            "oldInfo":{
                                "contour": puzzleCnt,
                                "imgHash": imgHash,
                            }
                        }
                    else:
                        iw.setNewSudoku()
                        imgSolveM.value = {
                            "iw":iw,
                            "oldInfo":{
                                "contour": puzzleCnt,
                                "imgHash": imgHash,
                            }
                        }
                except Exception as e:
                    # 没有找到矩形
                    imgSolveM.value = None
                    pass
                if(imgPutQ.qsize() < MAX_QUEUE_SIZE):
                    imgPutQ.put(iw)

        except Exception as error:#,Argument
            print("****************************",error)
            break
        # print('fp',image)

# 耗时的 低频的处理
def solveSudokuProcess(modelPath,imgSolveM,imgSolutionM):
    solveSudokuSerial = None
    solveCnt = 0
    model = load_model(modelPath)
    while 1:
        try:
            time.sleep(0.02)
            iwMap = imgSolveM.value
            iw = iwMap['iw']
            if(iw is None):
                continue
            if(solveSudokuSerial == iw.sudokuSerial):
                if( solveCnt > 1):
                    continue
            else:
                solveCnt = 0
            print(f"sudokuSerial:{iw.sudokuSerial} solveCnt:{solveCnt}")
            cellLocs, puzzleNp = analysis_pussle_image(
                iw.warped,
                iw.contour,
                model
            )
            puzzle = Sudoku(3,3,puzzleNp.tolist())
            solution = puzzle.solve()
            solutionStr = str(solution)
            if("SOLVED" in solutionStr):
                ow = imgSolutionM.value
                solveSudokuSerial = iw.sudokuSerial
                iw.cellLocs = cellLocs
                iw.puzzle = puzzle.board
                iw.solution = solution.board
                imgSolutionM.value = iw
                if(ow is not None and ow.sudokuSerial == iw.sudokuSerial and 
                    np.all( np.array( solution.board) == np.array(ow.solution))):
                    solveCnt+=1
                    time.sleep(0.3)
                else:
                    print("新题目或计算不等")
                    solveCnt = 0
                    time.sleep(0.3)
            else: # 求解失败 延时再解一次
                print(solutionStr.split("---------------------------")[1])
                puzzle.show()
                time.sleep(0.3)
                solveCnt = 0
        except Exception as error:#,Argument
            # print(f"**** ERROR: ****",error)
            print(f"**** {inspect.currentframe().f_code.co_name} ERROR: ****",error)
            et, ev, tb = sys.exc_info()
            msg = ''.join(traceback.format_exception(et, ev, tb))
            print(msg)

def putProcess(imgPutQ,imgSolutionM):
    buff = []
    buffSize = 5
    before = -1
    while True:
        try:
            iw = imgPutQ.get(timeout=10)
            if(not iw is None):
                pass # 插入帧
                if(len(buff)<=0):
                    buff.append(iw)
                else:
                    if(iw.serial == buff[0].serial-1 ):
                        buff.insert(0,iw)
                    elif(iw.serial > buff[0].serial):
                        for i in range(len(buff)):
                            if(iw.serial > buff[i].serial):
                                buff.insert(i+1,iw)
                                break
            
            if(len(buff) and buff[0].serial <= before+1 or len(buff) > buffSize):
                shim = buff.pop()
                before = shim.serial
                
                siw = imgSolutionM.value
                # draw
                if(siw and shim.sudokuSerial is not None and shim.sudokuSerial == siw.sudokuSerial ):
                    draw_sudoku_solution_on_src(
                        shim.img,
                        shim.contour,
                        shim.correctionImgShape,
                        siw.puzzle,
                        siw.solution
                    )
                elif(shim.contour is not None):
                    cv.drawContours(shim.img,[shim.contour],-1,(100,255,100),2)
                cv.imshow('pnp',shim.img)
                cv.waitKey(10)
        except Exception as error:#,Argument
            # print(f"**** ERROR: ****",error)
            print(f"**** {inspect.currentframe().f_code.co_name} ERROR: ****",error)
            et, ev, tb = sys.exc_info()
            msg = ''.join(traceback.format_exception(et, ev, tb))
            print(msg)
            # break # 让Pool返回就能正常退出了
    cv.destroyAllWindows()

if __name__ == '__main__':
    print('run program')
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
        help="path to trained digit classifier")
    ap.add_argument("-i", "--image", required=False,
        help="path to trained digit classifier")
    args = vars(ap.parse_args())

    modelPath = args["model"]

    # imgIntQ = Queue()
    # imgPutQ = Queue()
    imgIntQ = Manager().Queue() # mainProcess -> fcProcess
    imgPutQ = Manager().Queue() # fcProcess -> putProcess
    imgSolveM = Manager().Value(None,None) # fcProcess -> solveSudokuProcess
    imgSolutionM = Manager().Value(None,None) # putProcess -> solveSudokuProcess
    
    mp = Process(name="mainProcess",target=mainProcess, args=(imgIntQ,)) # 获取视频进程
    pp = Process(name="putProcess",target=putProcess, args=(imgPutQ,imgSolutionM)) # 显示结果进程
    ssp = Process(name="solveSudokuProcess",target=solveSudokuProcess, args=(modelPath,imgSolveM,imgSolutionM)) # 耗时的 低频的处理 识别和解算细节
    mp.start()
    pp.start()
    ssp.start()

    poolcnt = 1
    fpo = Pool(poolcnt) # 并行处理进程
    fpo.map(fcProcess, [(imgIntQ,imgPutQ,imgSolveM)]*poolcnt)

    mp.join()
    print("mp exit")

    imgIntQ.close()
    imgPutQ.close()
    imgSolveM.close()
    imgSolutionM.close()

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