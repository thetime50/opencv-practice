import numpy as np
def getDctTransform(length):
    coe = np.empty((length,length),dtype="float64")
    for k in range(length):
        for n in range(length):
            coe[k][n] = np.cos(k*np.pi*(n+0.5)/length)
    def dctTransform(wave):
        res = np.zeros([length],dtype="float64")
        for k,kcoe in enumerate(coe):
            resk = 0
            for n,ncoe in enumerate(kcoe):
                if(n<wave.shape[0]):
                    resk+=wave[n]*ncoe
                else:
                    break
            # resk = resk*np.sqrt(1/length) 
            # if(k):
            #     resk *=np.sqrt(2)
            res[k] = resk
        return res
    return dctTransform
# modulus: 显示的变换结果，比原本结果的长度小
# N: 原本数据长度
def getDctFun(modulus,N=None):
    if(N is None):
        N = len(modulus)
    def dctFun(x):
        res = 0
        for k, v in enumerate(modulus):
            # seta = x*np.pi*(k+0.5)/N
            # seta = 2*np.pi/(2*N-1) * k*x
            seta = np.pi * (x+0.5)*k/N
            v =  np.cos(seta) * v
            if(k):
                v*=2
            res += v
        return res
    return dctFun