import cv2 as cv
import numpy as np
import os
import time

print(True)


CAMERA_RUL = 'rtsp://admin:admin@192.168.1.148:8554/live'  # ip摄像头 #帧率不太对
# CAMERA_RUL = 'rtsp://admin:admin@192.168.31.60:8554/live'  # ip摄像头 #帧率不太对

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
    def __init__(self,img):
        self.timestamp = time.time()
        self.serial=ImgWrap.cnt
        ImgWrap.cnt+=1
        self.img = img

def mainProcess(imgIntQ):
    def init(pd):
        pass
    def ring(state,key,image):
        imgIntQ.put(ImgWrap(image))
        return state,image
    ## start
    rtmp_str = CAMERA_RUL
    # producer = Producer(rtmp_str,init=init,ring=ring)  # 开个线程
    producer = Producer(rtmp_str,'load',init=init,ring=ring)  # 开个线程
    producer.start()
    print("mpend")


def fcProcess(para):
    (imgIntQ,imgPutQ) = para
    while True:
        try:
            image = imgIntQ.get(timeout=300)
            if(image is None):
                print("fc queue is None")
                # break
        except Exception as error:#,Argument
            print("****************************",error)
            # break
        # print('fp',image)
        imgPutQ.put(image)

def putProcess(imgPutQ):
    buff = []
    buffSize = 5
    before = 0
    while True:
        try:
            iw = imgPutQ.get(timeout=300)
            if(iw is None):
                print("pp queue is None")
                break
        except Exception as error:#,Argument
            print("****************************",error)
            break
        # print(iw)
        # print("pp",iw.img.shape)
        # [3,4,5,6]
        if(len(buff)==0):
            buff.append(iw)
        else:
            for i in range(len(buff)):
                index = len(buff)-i-1
                if(iw.serial >= buff[index].serial):
                    buff.insert(index+1,iw)
                    break
        if (buff[0].serial <= before+1) or len(buff)>=buffSize:
            shim = buff.pop(0)
            before = shim.serial
            cv.imshow('pnp',shim.img)
            cv.waitKey(10)
        # im = np.full((30,30,3),200,np.uint8)
        # cv.imshow('pnp',im)
    cv.destroyAllWindows()

if __name__ == '__main__':
    print('run program')
    # imgIntQ = Queue()
    # imgPutQ = Queue()
    imgIntQ = Manager().Queue()
    imgPutQ = Manager().Queue()

    mp = Process(target=mainProcess, args=(imgIntQ,))
    pp = Process(target=putProcess, args=(imgPutQ,))
    mp.start()
    pp.start()

    poolcnt = 2
    fpo = Pool(poolcnt)
    fpo.map(fcProcess, [(imgIntQ,imgPutQ)]*poolcnt)

    mp.join()
    print("mp exit")

    imgIntQ.close()
    imgPutQ.close()

    # fpo.close()
    # fpo.terminate()
    fpo.join()
    print("fpo exit")

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