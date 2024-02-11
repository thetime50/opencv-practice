import matplotlib.pyplot as plt
import numpy as np
from IPython.core.display import HTML

# %matplotlib notebook

styltStr = '''
<style>
    div.output_scroll{
    height: unset;
    }
</style>
'''
HTML(styltStr)

# define
dataLen = 128

import fourier

try:
    squareWave = np.load("./data/square_wave.npz")
    rightTriangleWave = np.load("./data/right_triangle_wave.npz")
    isoscelesTriangleWave = np.load("./data/isosceles_triangle_wave.npz")
    stepWave = np.load("./data/step_wave.npz")
    triFunWave = np.load("./data/trifun_wave.npz")
except Exception as e:
    print("打开数据失败,请尝试生成数据",e)


# def showFft(title,y_origin,showrange=None,showwave=True,showNpfft = False):
#     x_values = np.arange(0,len(y_origin))
#     if(showrange is None):
#         showrange = len(y_origin)
#     akArr = fourier.ditFftFloat(y_origin)
#     y=0
#     if(showwave):
#         ft = fourier.getDftFun(akArr[:showrange],len(akArr))
#         wave = np.vectorize(ft)(x_values)
#         plt.subplot2grid((2,2),(y,0),colspan=2)
#         y+=1
#         plt.plot(x_values, y_origin, label='origin')
#         plt.plot(x_values, wave, label='fft'+str(showrange))
#         plt.legend()

#     modulus = fourier.getModulus(akArr)
#     phase = fourier.getPhase(akArr)
#     plt.subplot2grid((2,2),(y,0),colspan=2)
#     y+=1
#     plt.plot(x_values, modulus, label='modulus')
#     plt.plot(x_values, phase, label='phase')
#     plt.legend()

#     if(showNpfft):
#         akArr1 = np.fft.fft(y_origin)
#         modulus1 = fourier.getModulus(akArr1) / len(akArr1)
#         phase1 = fourier.getPhase(akArr1)
#         plt.subplot2grid((2,2),(y,0),colspan=2)
#         y+=1
#         plt.plot(x_values, modulus1, label='modulus')
#         plt.plot(x_values, phase1, label='phase')
#         plt.legend()
#     plt.title(title,y=2.2 if y>1 else 1.1)
#     plt.show()

# def showAveFft(title,y_origin,showrange=None):
#     y_origin = y_origin - np.average(y_origin)
#     showFft(title+"_ave",y_origin,showrange,showwave = False,showNpfft=True)



# showFft("step_wave",stepWave["data"],60)
# showAveFft("step_wave",stepWave["data"],60)
# showFft("trifun_wave",triFunWave["data"],60)
# showAveFft("trifun_wave",triFunWave["data"],60)
    
    
import fourier

import importlib;
importlib.reload(fourier)
t = None
def showDct(title,y_origin,showrange=None,showwave=True):
    x_values = np.arange(0,len(y_origin))
    if(showrange is None):
        showrange = len(y_origin)
    dctFloat = fourier.getDctTransform(len(y_origin))
    modulus = dctFloat(y_origin)/len(y_origin)

    if(showwave):
        ft = fourier.getDctFun(modulus[:showrange],len(modulus))
        wave = np.vectorize(ft)(x_values)
        plt.subplot2grid((2,2),(0,0),colspan=2)
        plt.plot(x_values, y_origin, label='origin')
        plt.plot(x_values, wave, label='dct'+str(showrange))
        plt.legend()

    plt.subplot2grid((2,2),(1 if showwave else 0 ,0),colspan=2)
    plt.plot(x_values, modulus, label='modulus')
    plt.legend()
    plt.title(title,y=2.2 if showwave else 1.1)
    plt.show()
    return modulus

def showAveDct(title,y_origin,showrange=None):
    y_origin = y_origin - np.average(y_origin)
    showDct(title+"_ave",y_origin,showrange,showwave = False)


t = showDct("step_wave",stepWave["data"])#,60)
showAveDct("step_wave",stepWave["data"])#,60)
showDct("trifun_wave",triFunWave["data"])#,60)
showAveDct("trifun_wave",triFunWave["data"])#,60)
    

from scipy.fftpack import dct, idct
import numpy as np

# 原始信号
signal = stepWave["data"]

# 进行 DCT 变换
dct_coefficients = dct(signal)/len(signal)/2
reconstructed_signal = idct(dct_coefficients)
# reconstructed_signal[0 ]= 0

x_values = np.arange(0,len(signal))
plt.subplot2grid((1,2),(0,0),colspan=2)
plt.plot(x_values, signal,label = 'origin')
plt.plot(x_values, dct_coefficients,label = 'dct')
plt.plot(x_values, reconstructed_signal,label = 'reconstructed')
plt.legend()
plt.show()

plt.plot(x_values, dct_coefficients,label = 'dct')
plt.plot(x_values, t,label = 'my dct')
plt.legend()
plt.show()
pass