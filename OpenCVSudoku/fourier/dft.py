import numpy as np

def dftFloat(wave,cnt=None):
    if(cnt is None):
        cnt = len(wave)
    akArr = []
    N = len(wave)
    for k in range(len(wave)):
        ak = np.complex64(0)
        for n in range(len(wave)):
            seta = 2*np.pi/N * n*k
            ak += wave[n] * np.complex64(np.cos(seta)+ np.sin(seta)*1.0j)
        akArr.append(ak/N)
    return akArr


def getDftFun(akArr,N=None):
    if(N is None):
        N = len(akArr)
    def dftFun(x):
        res = 0
        for i, v in enumerate(akArr):
            seta = 2*np.pi/N * i*x
            res += np.cos(seta)*v.real + np.sin(seta)*v.imag
        return res
    return dftFun

def getModulus(akArr):
    res = np.empty([len(akArr)],"float64")
    for i,ak in enumerate( akArr):
        res[i] = np.sqrt(ak.real**2 + ak.imag**2)
    return res

def getPhase(akArr):
    res = np.empty([len(akArr)],"float64")
    for i,ak in enumerate( akArr):
        if(ak.real != 0):
            res[i] = np.arctan(- ak.imag/ak.real)
        else:
            res[i] = np.arctan(- ak.imag*np.inf)
    return res

