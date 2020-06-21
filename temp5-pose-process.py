import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tool
import pylab
import os
import time

# %matplotlib inline
# pylab.rcParams['figure.figsize'] = (15.0,7.0) #调整显示大小
# tool.setGrid(1,2)

print(True)



import threading
from multiprocessing import Process, Queue

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
        

class FindCorners():
    def __init__(self,shape):
        self.findCnt= 0
        self.objpoints=[]
        self.imgpoints=[]
        self.lastCornersRet = False
        self.lastCorners = None
        self.shape = shape
        self.objp = np.zeros((self.shape[0]*self.shape[1],3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.shape[0],0:self.shape[1]].T.reshape(-1,2)
        self.criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.imgShape = None
        
        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = 0,None,None,None,None,
        self.newcameramtx, self.roi = None,None
    def load(self,file):
        load = np.load(file)
        # print(dir(load),load.files)

        self.ret = load['arr_0']
        self.mtx = load['arr_1']
        self.dist = load['arr_2']
        self.rvecs = load['arr_3']
        self.tvecs = load['arr_4']
        self.newcameramtx = load['arr_5']
        self.roi = load['arr_6']
        print("loaded file from",file,'\n', self.ret,'\n',self.newcameramtx,self.roi)
    def save(self,file=''):
        if(not file):
            file = './doc/temp3-calibrate-'+time.strftime("%Y-%m-%d %H%M%S", time.localtime())
        np.savez(file,
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs,
            self.newcameramtx, self.roi)
        print('saved para to',file)
    def find(self,img):
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.imgShape= gray.shape[::-1]
        self.lastCornersRet, self.lastCorners = cv.findChessboardCorners(gray, self.shape, None)
        self.findCnt+=1
        if(self.lastCornersRet):
            self.objpoints.append(self.objp)
            corners2 = cv.cornerSubPix(gray,self.lastCorners, (11,11), (-1,-1), self.criteria)
            self.imgpoints.append(corners2)
        print("state: find:%d finded:%d fail:%d"%(self.findCnt,len(self.imgpoints),self.findCnt-len(self.imgpoints)),
            end='\r\r\r')
        return self.lastCornersRet, self.lastCorners
        # cv.drawChessboardCorners(det, self.shape, corners2, ret)
        # cv.imshow('finded', det)
    def drawChessboardCornersLast(self,img):
#         if(self.lastCornersRet):
        cv.drawChessboardCorners(img, self.shape, self.lastCorners,  self.lastCornersRet)

    def calibrate(self):
        if(len(self.imgpoints)):
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = \
                cv.calibrateCamera(self.objpoints, self.imgpoints, self.imgShape, None, None)
            self.newcameramtx, self.roi = \
                cv.getOptimalNewCameraMatrix(self.mtx, self.dist, self.shape, 1,self.shape)
        else:
            self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = 0,None,None,None,None,
            self.newcameramtx, self.roi = None,None
        return self.ret,self.newcameramtx, self.roi
    def undistort(self,img):
        # print(self.roi, self.roi and self.roi[2] , self.roi and self.roi[3])
        if(self.roi.any() and self.roi[2] and self.roi[3]):
            return cv.undistort(img, self.mtx, self.dist, None, self.newcameramtx)
        else:
            return np.full_like(img,180)

    def draw(self, img, corners, imgpts):
        corner = tuple(corners[0].ravel())
        img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
        img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
        img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
        return img
    def drawCube(self, img, corners, imgpts):
        imgpts = np.int32(imgpts).reshape(-1,2)
        # draw ground floor in green
        img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
        # draw pillars in blue color
        for i,j in zip(range(4),range(4,8)):
            img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
        # draw top layer in red color
        img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
        return img
    def solvePnP(self,img):
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray,self.shape,None)#(6,7)
        if ret == True:
            axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
            criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)#self.criteria)
            # print("****",corners2.reshape((len(corners2),2)))
            # Find the rotation and translation vectors.
            # print("*****",self.objp, corners2, self.mtx, self.dist)
            print("*****",self.objp.shape, corners2.shape, self.mtx.shape, self.dist.shape)
            ret,rvecs, tvecs = cv.solvePnP(self.objp, corners2, self.mtx, self.dist)
            # ret,rvecs, tvecs = cv.solvePnPRansac(self.objp, corners2, self.mtx, self.dist)
            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, self.mtx, self.dist)
            img = self.draw(img,corners2,imgpts)
            print(img.shape)
            # cv.imshow('pnp',img)
            return img
        else:
            print("corners flase",end="\n")
           
class ImgWrap:
    def __init__(self,img):
        self.timestamp = time.time()
        self.img = img

def mainProcess(imgIntQ):
    def init(pd):
        pass
    def ring(state,key,image):
        imgIntQ.put(ImgWrap(image))
        return state,image
    ## start
    rtmp_str = 'rtsp://admin:admin@192.168.1.154:8554/live'  # ip摄像头 #帧率不太对
    # rtmp_str = 'rtsp://admin:admin@192.168.31.60:8554/live'  # ip摄像头 #帧率不太对
    # producer = Producer(rtmp_str,init=init,ring=ring)  # 开个线程
    producer = Producer(rtmp_str,'load',init=init,ring=ring)  # 开个线程
    producer.start()


def fcProcess(imgIntQ,imgPutQ):
    fc =None
    fc=FindCorners((7,7))
    file = "./doc/temp3-calibrate-2020-06-16 214523.npz"
    fc.load(file)
    while True:
        image = imgIntQ.get()
        imgPutQ.put(fc.solvePnP(image))

def putProcess(imgPutQ):
    while True:
        iw = imgPutQ.get()
        cv.imshow('pnp',iw.img)

if __name__ == '__main__':
    print('run program')
    imgIntQ = Queue()
    imgPutQ = Queue()
    mp = Process(target=mainProcess, args=(imgIntQ,))
    # mp = Process(target=mainProcess, args=(imgIntQ,))
    # fp = Process(target=fcProcess, args=(imgIntQ,imgPutQ))
    # pp = Process(target=putProcess, args=(imgPutQ,))

    mp.start()
    # fp.start()
    # pp.start()


    mp.join()
    print(444)
    # fp.close()
    # pp.close()
    # fp.join()
    # pp.join()

    cv.destroyAllWindows()
    print("all end.....")

    
# close()跟terminate()的区别在于close()会等待池中的worker进程执行结束再关闭pool,而terminate()则是直接关闭