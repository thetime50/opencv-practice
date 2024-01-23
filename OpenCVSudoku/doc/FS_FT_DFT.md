
## 和差化积
- [百度百科 和差化积](https://baike.baidu.com/item/%E7%A7%AF%E5%8C%96%E5%92%8C%E5%B7%AE/6973123)
- [知乎 和差化积的几何直觉](https://zhuanlan.zhihu.com/p/184757814)

**积化和差**  
![sina_cosb](./img/sinacosb.svg)  
![cosa_sinb](./img/cosasinb.svg)  


![cosa_cosb](./img/cosacosb.svg)  
![sina_sinb](./img/sinasinb.svg)  


异积 = s和+-s差  
同积 = c和+负s差

**和差化积**  

![sina+b](./img/sina+b.svg)  
![sina-b](./img/sina-b.svg)  

![cosa+b](./img/cosa+b.svg)  
![cosa-b](./img/cosa-b.svg)  

s和差 = 异积+-异积  
c和差 = c积-+s积


## 傅里叶级数
- [傅里叶级数和傅里叶变换是什么关系？](https://www.zhihu.com/question/21665935)
- [傅里叶系列（一）傅里叶级数的推导](https://zhuanlan.zhihu.com/p/41455378)

周期函数能够通过傅里叶级数画出频域图  
周期T=∞时频域图变为连续的曲线，引入复函数e，得到傅立叶变换

傅里叶公式  
*-  
f(x) 为待分析(拟合)的函数  
$A_0$ 为直流分量  
$A_n$ 为各频率下的三角函数参数(振幅 相位)  
n 为频率倍数  
$\omega$ 为角速度 ($\omega = \frac{2 \pi}{T}$)  
t 为函数变量  
$\psi_n$ 为各频率时的相位  
-*
$$ 
\begin{align}
f(t) &= A_0 + \sum_{n=1}^{\infty }A_{n}sin(n\omega t+\psi _{n})\\
&=  A_0 + \sum_{n=1}^{\infty }A_{n}(sin(n\omega t)cos(\psi_n) + cos(n\omega t)sin(\psi_{n}))
\end{align} 
$$
令 $a_n=An\cdot sin\psi_n,b=A_n\cdot cos\psi_n$得到 **(6)**
$$
f(t) = A_0 + \sum_{n=1}^{\infty}[a_n cos(n \omega t) + b_n sin(n\omega t)]
$$

泰勒级数 麦克劳林 原理
$$
f(x) = A + Bx + Cx^2 + Dx^3 ... \\
f'(x) = B + 2Cx + 3Dx^2... \\
f''(x) = 2C + 6Dx + ... \\

逐次对函数求导给未知数降幂 \\
把x=0带入傅里叶级数的函数和原函数 就可以把每一级系数ABC求出来\\
原理 通过求导拆解出每一项的系数，通过x=0把后面项清空\\
A = f(0) \\
B = f'(0) \\
C = {f}''(0) /2 \\
D = {f}'''(0) /(1*2*3) \\
$$

<br/>
在傅里叶级数中 通过 

### 计算$A_0$
把(6)式在 $n\omega t$为-Π到Π的区间内积分，
$$
\begin{align}
\int_{-\pi}^{\pi}f(t) &= \int_{-\pi}^{\pi}A_0 + \int_{-\pi}^{\pi}A_0\sum_{m=1}^{\infty}[a_ncos(n\omega t) + b_n sin(n\omega t)] \\
&=2\pi A_0
\end{align}
\\
\therefore A_0 = \frac{1}{2\pi} \int_{-\pi}^{\pi}f(t)
$$

### 计算$a_n b_n$

1. 和差化积公式 
2. -Π到Π区间正交函数积分为0来消除项  
  (k)

把(6)式两边同时乘以 cos(k&omega;t) 然后积分  
同乘 $cos(k\omega t)$
$$
\begin{align}
f(t)\cdot cos(k\omega t) =& A_0 \cdot cos(k\omega t) + \\ 
& \sum_{n=1}^{\infty}[a_n cos(n \omega t) \cdot cos(k\omega t) + b_n sin(n\omega t) \cdot cos(k\omega t)]
\end{align}
$$

积分 并且令k=n时只有蓝色项不为0
$$
\begin{align}
\int_{-\pi}^{\pi}f(t) \cdot cos(k\omega t) dt &= A_0{\color{Red} \int_{-\pi}^{\pi} cos(k\omega t) dt} + \\
&\sum_{m=1}^{\infty}[a_n{\color{Blue}\int_{-\pi}^{\pi}cos(n\omega t) \cdot cos(k\omega t) dt}  + b_n {\color{Red} \int_{-\pi}^{\pi} sin(n\omega t) \cdot cos(k\omega t) dt }] \\
&= a_n \int_{-\pi}^{\pi} cos^2(n\omega t)dt\\
&= \frac{a_n}{2}\int_{-\pi}^\pi(1+cos2n\omega t) dt \qquad\text{(半角公式)}\\
&=\frac{a_n}{2}(\int_{-\pi}^\pi 1dt + {\color{Red} \int_{-\pi}^\pi cos(2n\omega t) dt})\\
&=a_n\pi
\end{align}
\\
\\
\therefore a_n = \frac{1}{\pi}\int_{-\pi}^{\pi}cos(n\omega t)\cdot f(t)dt \quad (k=n)
$$

同理得
$$
\therefore b_n = \frac{1}{\pi}\int_{-\pi}^{\pi}sin(n\omega t)\cdot f(t)dt \quad (k=n)
$$
<div id="ft"></div>

令 a_0 = 2A_0 假设T = 2Π  
最终
$$
\begin{align}
f(t) &= a_0/2 + \sum_{n=1}^\infty[a_n cos(n\omega t) + b_n sin(n\omega t)] \\
a_0 &= \frac{2}{T}\int_{-\pi}^{\pi}f(t)dt\\
a_n &= \frac{2}{T}\int_{t_0}^{t_0+T}f(t)cos(n\omega t)dt \\
b_n &= \frac{2}{T}\int_{t_0}^{t_0+T}f(t)sin(n\omega t)dt \\
\end{align}
$$

## 傅里叶变换推导详解
[傅里叶变换推导详解](https://zhuanlan.zhihu.com/p/77345128)  
(2.14)  
波幅 相位与傅里叶级数$a_n \quad b_n$的关系
$$
\begin{align}
c_n &= \sqrt{{a_n}^2+{b_n}^2} \\
\varphi &= arctan(-\frac{b_n}{a_n})
\end{align}
$$

#### 复变函数到傅里叶级数
引入复数函数表达式：(和欧拉公式有关?)  
$e^{j\theta} = cos\theta + j sin\theta $  
又 $\theta = \omega t = \frac{2\pi}{T}t $  
得 $e^{j\omega t} = cos(\omega t) + jsin(\omega t)$  
设一组三角函数 频率是$cos\omega t$的整数n倍,则这些三角函数为  
$$
\begin{align}
cos(n\omega t) &= \frac{e^{jn\omega t} + e^{-jn\omega t}}{2} \\
sin(n\omega t) &= \frac{e^{jn\omega t} - e^{-jn\omega t}}{2j}
\end{align}
$$
带入傅里叶级数得
$$
\begin{align}
f(t) &= c_0 + \sum_{n=1}^{\infty}[a_n\frac{e^{jn\omega t} + e^{-jn\omega t}}{2} + b_n\frac{e^{jn\omega t} - e^{-jn\omega t }}{2j}] \\ 
f(t) &= c_0 + \sum_{n=1}^{\infty}[\frac{(a_n-jb_n)}{2}e^{jn\omega t}+\frac{(a_n + jb_n)}{2}e^{-jn\omega t}]
\end{align}
$$
$a_n\quad b_n$是三角函数的积分所以
$$
a_{-n} = a_n\\
b_{-n} = b_n
$$
带入可得
$$

f(t) = c_0 + \sum_{n=1}^{\infty}[\frac{(a_n-jb_n)}{2}e^{jn\omega t}+\frac{(a_{-n} + jb_{-n})}{2}e^{-jn\omega t}]\\
f(t) = c_0 + \sum_{n=1}^{\infty}\frac{(a_n-jb_n)}{2}e^{jn\omega t}+\sum_{n=-1}^{-\infty}\frac{(a_{n} - jb_{n})}{2}e^{jn\omega t}
$$
$c_0$即为n=0的情况， 所以
$$
f(t) = \sum_{n=-\infty}^\infty \frac{(a_n-jb_n)}{2}e^{jn\omega t}
$$
设 $A_n = \frac{(a_n-jb_n)}{2}$得  
$f(t) = \sum_{n=-\infty}^\infty A_n e^{jn\omega t}$

**现在要去除求和符号把$A_n$提出来，通过代数计算做变换**

两边同乘以$e^{-jk\omega t}$并积分 
$$
\int_0^Tf(t)e^{-jk\omega t}dt = \int_0^T\sum_{n=-\infty}^{\infty}A_ne^{j(n-j)\omega t}dt
$$
带入三角函数正交积分为零推论，此时右边只有n=k 时有值

$$
\int_0^Tf(t)e^{-jn\omega t}dt = A_nT \\
\therefore 得 A_n = \frac{1}{T}\int_0^Tf(t)e^{-jn\omega t}dt \\
幅值 \left|A_n\right| = \frac{1}{2}\sqrt{{a_n}^2 + {b_n}^2 } = \frac{1}{2}c_n
$$

#### 周期性离散时间傅里叶变换

下面的$\Sigma$下标n k t描述可能不对

设离散时间的采样样本为 x[t],其周期为T,那么,其应该频率是 $\frac{2\pi}{T}$ 有公式描述：  
$x[t] = x[t+kT]$  

设经历时间$t = <T>$即一个周期内的采样点，则周期离散傅里叶级数可写成  
$x[t] = \sum_{n=<T>}A_ne^{jn\omega t} \quad(3.36)$  

两边同乘以$e^{-jk\omega t}$ (傅里叶公式中加入采样的函数描述，k来自采样 n来自傅里叶)  
$x[t]e^{-jk\omega t} = \sum_{n=<T>}A_ne^{jn\omega t}e^{-jk\omega t}$  

两边同时进行T项上的求和  
$\sum_{t=<T>}x[t]e^{-jk\omega t} = \sum_{t=<T>}\sum_{n=<T>}A_ne^{j(n-k)\omega t}$  

上式同样满足当n不等于k时,周期的累加和为0,因此,上式可变为  
$\sum_{t=<T>}x[t]e^{-jk\omega t} = T A_n$  

<div id="dft"></div>

最终得 $A_n = \frac{1}{T}\sum_{t=<T>}x[t]e^{-jn\omega t} \quad(3.39)$  
**$A_n$** n为某个频率时的分解参数
**t** 为固定间隔的采样时间点  
**n** 为角频率倍数

#### 非周期离散时间傅里叶变换
让他在周期时间属于$[-\infty,\infty]$即可认为是非周期时间无限的函数，则各频率下的系数计算为

$A_n = \frac{1}{N}\sum_{t=-\infty}^{\infty}x[t]e^{-jn\omega t}$

设频谱 $X(e^{j\omega}) = \sum_{t=-\infty}^{\infty}x[t]e^{-j\omega t}$  
则 $A_n = \frac{1}{N}X(e^{j\omega n})$  
再带回采样表达式 $x[t] = \sum_{n=<T>}A_nd^{jn\omega t}$ 得  
$x[t] = \frac{1}{N}\sum_{t=-\infty}^{\infty}x(e^{j\omega t})e^{jn\omega t}$  

又因为 $N = \frac{2\pi}{\omega}$
$x[t] = \frac{\omega}{2\pi}\sum_{t=-\infty}^{\infty}x(e^{j\omega t})e^{jn\omega t}$  

因为周期无穷大，$\omega$无穷小，所以上式变为了$j\omega$于$[0,2\pi]$内得积分,得  
$x[t] = \frac{1}{2\pi}\int_0^{2\pi}X(e^{j\omega})e^{j\omega n t}d\omega$


#### 非周期有限长度离散时间傅里叶变换

根据[DFT](#dft)的$A_n、x[t]$公式  
$x[t] = \sum_{n=<T>}A_ne^{jn\omega t} \quad(3.36)$  
$A_n = \frac{1}{T}\sum_{t=<T>}x[t]e^{-jn\omega t} \quad(3.39)$

带入t范围为[0,T-1],并带入$\omega=\frac{2\pi}{T}$
$$
\begin{align}
x[t] &= \sum_{n=0}^{T-1}A_n e^{jn\omega t} \\
    &=\sum_{n=0}^{T-1}A_ne^{jn\frac{2\pi}{T}t} \quad (3.47)
\end{align}
$$
同理得
$$
A_n=\frac{1}{T}\sum_{t=0}^{T-1}x[t]e^{-jn\frac{2\pi}{T}t} \quad(3.49)
$$
常常用大写的$X[n]$来表示变换后的复信号的T倍(即求其频率密度),即
$X[n] = TA_n$  

那么3.47 3.49可改写为
$$
x[t] =\frac{1}{T}\sum_{n=0}^{T-1}X[n]e^{jn\frac{2\pi}{T}t} \quad (3.51)
$$

$$
X[n]=\sum_{t=0}^{T-1}x[t]e^{-jn\frac{2\pi}{T}t} \quad(3.52)
$$

#### 注意
离散福利叶变换的幅度计算和相位计算与傅里叶级数的余弦展开(2.14)有所不同  
根据欧拉公式 $e^{-j\omega t} = cos(\omega t) - j sin(\omega t)$与傅立叶级数余弦展开角度+-号不同
$$
\begin{align}
c_n &= \frac{2}{T}\sqrt{{a_n}^2+{b_n}^2} \\
\varphi &= arctan(\frac{b_n}{a_n})
\end{align}
$$

// todo 这个-号和2/T是哪里来的，应该是哪里得定义不一样

#### note
对于DFT $A_n = \frac{1}{T}\sum_{t=<T>}x[t]e^{-jn\omega t} \quad(3.39)$
- n = 0 时对应的是傅立叶直流分量
- t = 0 时对应到不同频率n上可以用于在不同频率补齐到相位上

