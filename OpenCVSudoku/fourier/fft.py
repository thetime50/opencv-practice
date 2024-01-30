import numpy as np

class Euler:
    def __init__(self,N,):
        self.N = N
    def get(self,n,k):
        phi = 2*np.pi/self.N*n*k
        return np.cos(phi) + np.sin(phi)*1j

def exp2complex(N,n,k):
    phi = 2*np.pi/N * n*k
    return np.cos(phi) + np.sin(phi)*1j


def ditFftFloat_(wave,_indexArr):
    N = len(_indexArr)
    halfN = N//2
    Xk=np.empty(_indexArr.shape,dtype=np.complex64)
    if(len(_indexArr) > 2):
        X1k = ditFftFloat_(wave,_indexArr[::2])/2
        X2k = ditFftFloat_(wave,_indexArr[1::2])/2
        for k in range(halfN):
            X2k_ = X2k[k] * exp2complex(N,1,k)
            Xk[k] = X1k[k] + X2k_
            Xk[k+halfN] = X1k[k] - X2k_
    else:
        exp2complex(N,1,0)
        w = wave[_indexArr]
        X1 = w[0]* exp2complex(N,0,0)
        X2 = w[1]* exp2complex(N,1,0)
        Xk[0] = (X1 + X2)/2
        Xk[1] = (X1 - X2)/2
    return Xk
def ditFftFloat(wave,cnt=None):
    if(cnt is None):
        cnt = len(wave)
    cnt = int(np.power(2, np.ceil( np.log2(cnt))))
    if(len(wave)!=cnt):
        paddingSize = cnt - len(wave)
        wave = np.pad(wave, [0,paddingSize],mode="constant")
    return ditFftFloat_(wave,np.arange(cnt))
    


