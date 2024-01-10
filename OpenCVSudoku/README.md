# opencv-sudoku-solver-and-ocr

https://www.pyimagesearch.com/2020/08/10/opencv-sudoku-solver-and-ocr/

## 文件说明

- solve_sudoku_puzzle.py 单张图片数独求解
- solve_sudoku_stream.py 视频流数独求解
- train_digit_classifier.py 视频流数独求解 使用进程池 还没添加解析
- train_digit_classifier2.py 视频流数独求解 单线程循环处理
- train_digit_classifier3.py 视频流数独求解 使用 RTSCapture 求解部分可优化
- RTSCapture 使用子线程循环读取cv.VideoCapture视频帧

- make_print_dataset.py 创建打印数数字据集 混合数字数据集
- train_digit_classifier.py 训练数字识别分类模型

## Sudokunet.py
[file->](.\pyimagesearch\models\SudokuNet.py)

- keras.layers.Conv2D

https://keras.io/api/layers/convolution_layers/convolution2d/  
https://keras-cn.readthedocs.io/en/latest/layers/convolutional_layer/  

2D卷积,
```python
Conv2D(32,(5,5),padding="same", # 过滤器 窗口 边缘处理
            input_shape=inputShape) # 输入形状
```
卷积层用来提取局部(窗口)特征,一个过滤器即提取识别一种特征的一个Neural  
有几个filter就可以在同一个区域得到多少个纹理(特征)结果，也就是几个channel,也就是图片的(长,宽,(深度))  
对于处理3*3像素的Neural 
如果输入的是灰度图，输入的数据为1的深度，也就是1个channel，输入9个数据
如果输入的是rgb图，输入的数据就有3的深度，也就是3个channel，输入为27个数据

```python
Activation("relu") # 激活函数
```
https://keras.io/zh/activations/

```python
MaxPooling2D(pool_size=(2,2)) # 池化 缩小比例
```
https://keras.io/zh/layers/pooling/#maxpooling2d

```python
Flatten()  # 展平成1维
```
https://keras.io/zh/layers/core/#flatten

```python
Dense()  # 展平成1维
```
https://keras.io/zh/layers/core/#dense

```python
Dropout()  # 就是Dropout 随机抽取节点排除 助于防止过拟合
```
https://keras.io/zh/layers/core/#dropout


## train_digit_classifier.py
[file->](.\rain_digit_classifier.py)

```python
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True, # 从 sys.argv 解析参数
	help="path to output model after training") # --help显示帮助信息
```

```python
le = LabelBinarizer()
le.fit([1,2,5,7,3]) # 枚举标签 要相同类型的数据好像
print(le.classes_)
le.transform([3,7]) # 将标签转为二进制flag表

trainLabels = le.fit_transform(trainLabels) # 注册标签枚举并转换为二进制表
```
https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelBinarizer.html

```python
opt = Adam(lr = INIT_LR) # adam优化器 RMSProp+Momentum(势能) 不记得了呀
model = SudokuNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy", # 编译模型 # 损失函数用交叉熵
    optimizer=opt, # 优化器
    metrics=["accuracy"])#训练结束后显示的评估数据
```
https://keras.io/zh/optimizers/#adam  
https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile

```python
print("[INFO] training network...")
H = model.fit( # 训练模型
	trainData, trainLabels,
	validation_data=(testData, testLabels), # 验证数据
	batch_size=BS, # 计算多少个样本的误差做一次模型更新
	epochs=EPOCHS, # 超参数 所有数据训练几轮
	verbose=1) # 进度条模式
```
https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit  
https://blog.csdn.net/weixin_42137700/article/details/84302045

```python

```
https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html


## puzzle.py
[file->](.\pyimagesearch\Sudoku\puzzle.py)

```python
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray,(7,7),3) # 高斯模糊

    thresh = cv2.adaptiveThreshold(blurred,255, # 自动阈值
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)# 高斯权重
    thresh = cv2.bitwise_not(thresh)
```
https://docs.opencv.org/3.4/d7/d1b/group__imgproc__misc.html#ga72b913f352e4a1b1b397736707afcde3


https://github.com/jrosebr1/imutils/blob/master/imutils/convenience.py