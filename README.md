# opencv-practice
[\[一起来撸\]OpenCV最新4.1.1官网直击](https://www.bilibili.com/video/BV1jJ411M7Bo)

[OpenCV-Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

tool:
- [numpy](https://numpy.org/devdocs/reference/index.html)

## TOC


## info

![pose.gif](./doc/pose1.gif)

## 环境
**2.10**  
conda create --name tf-cv imutils tensorflow=2.10 opencv pandas numpy matplotlib ipykernel

conda install --name tf-cv scikit-learn scikit-image  
pip install py-sudoku  

conda env create -f tf-cv.yml

**conda gpu**

https://blog.csdn.net/xuchaoxin1375/article/details/129698338

Note: Do not install TensorFlow with conda. It may not have the latest stable version. pip is recommended since TensorFlow is only officially released to PyPI.


conda create --name tf-cv python=3.9
conda activate tf-cv
<!-- conda search tensorflow -->
python -m pip install --upgrade pip
pip index versions tensorflow
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.10
conda install imutils opencv pandas numpy matplotlib ipykernel
conda install scikit-learn scikit-image  
pip install py-sudoku  

# 添加到jupyter内核选项
python kernel install --user --name tf-cv --display-name "tf-cv"


## conda 更新
```cmd
conda update anaconda-navigator
y
conda update anaconda-client
y
conda update -f anaconda-client
y
conda update navigator-updater
y
```

## tf-gpu win最后支持版本
tf2.10是最后支持win gpu的版本 之后的在wsl2上安装
[2.10 last version to support native Windows GPU](https://discuss.tensorflow.org/t/2-10-last-version-to-support-native-windows-gpu/12404)

## 在wsl中搭建tf-gpu环境

[安装miniconda](https://docs.conda.io/projects/miniconda/en/latest/)
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

```

[【Linux】conda: command not found解决办法](https://blog.csdn.net/weixin_38705903/article/details/86533863)

vim ~/.bashrc
在最后一行加上
export PATH=$PATH:/home/thetime/miniconda3/bin
保存后运行
source ~/.bashrc
测试
conda info --envs


### linux 卸载conda
[Linux卸载Anaconda](https://blog.csdn.net/hang916/article/details/79530108)