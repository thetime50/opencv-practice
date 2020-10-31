
# Holds the SudokuNet CNN architecture implemented with TensorFlow and Keras.
# tensorflow Keras CNN 字符识别

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout



class SudokuNet:
    @staticmethod
    def build(width,height,depth,classes):
        model = Sequential()
        inputShape = (height,width,depth)

		# first set of CONV => RELU => POOL layers
        model.add(Conv2D(32,(5,5),padding="same", # 过滤器 窗口 边缘处理
            input_shape=inputShape))
        model.add(Activation("relu")) # 激活函数 线性整流函数 (就是折线 水平线加一次函数)
        model.add(MaxPooling2D(pool_size=(2,2))) # 池化 缩小比例

		# second set of CONV => RELU => POOL layers
        model.add(Conv2D(32,(5,5),padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2)))

		# first set of FC => RELU layers
        model.add(Flatten()) # 展平成1维
        model.add(Dense(64)) # 全连接层
        model.add(Activation('relu'))
        model.add(Dropout(0.5)) # 就是Dropout 随机抽取节点排除 助于防止过拟合

		# second set of FC => RELU layers
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

		# softmax classifier
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model



