import os
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers,regularizers
from .const import IMG_SIZE
from .build_model import build_model

def build_model_pre():
    """使用预训练的MobileNetV2"""
    # os.environ['http_proxy'] = 'http://127.0.0.1:10908'
    # os.environ['https_proxy'] = 'http://127.0.0.1:10908'
    # https://blog.csdn.net/weixin_44519481/article/details/110006997
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # 冻结预训练层
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Concatenate()([inputs, inputs, inputs])  # 灰度图复制为3通道
    # x = layers.Conv2D(3, (1, 1), activation='relu')(inputs)  # 1通道转3通道
    # x = base_model.output
    x = base_model(x)

    # 叠加旧结构
    # 余弦退火
    # dataset 100 Epoch 3/20 到达 0.90
    # dataset 8000 Epoch 3/20 到达
    # 指数衰减
    # dataset 100 Epoch 20+n/20 到达 0.3337
    # dataset 8000 Epoch 20/20 到达 0.48 连续训练达到 loss 0.43 val_loss: 0.5654
    # outputs,has_sudoku,keypoints = build_model(x)
    # model = models.Model(inputs, outputs)
    # return model

    # 重新实现 用更小的网络
    # 余弦退火
    # dataset 100 Epoch 3/20 到达 0.29
    # dataset 8000 Epoch 3/20 到达 1.49
    # 指数衰减
    # dataset 100 Epoch /20 到达 
    # dataset 8000 Epoch 40/40 到达 0.3213 连续训练达到 loss 0.3077 val_loss: 1.3392
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    global_features = layers.GlobalAveragePooling2D()(x)

    # 是否有数独的分类头
    classification = layers.Dense(128, activation='relu')(global_features)
    classification = layers.Dropout(0.3)(classification)
    classification = layers.Dense(64, activation='relu')(classification)
    classification_output = layers.Dense(1, activation='sigmoid', name='has_sudoku')(classification)
    
    # 关键点回归头 (16个点 * 2坐标 = 32个值)
    regression = layers.Dense(256, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)
    regression = layers.Dense(128, activation='relu')(regression)
    regression = layers.Dense(64, activation='relu')(regression)
    regression_output = layers.Dense(32, activation='sigmoid', name='keypoints')(regression)  # 归一化坐标
    
    # 拼接成一个整体输出 (方便自定义损失)
    outputs = layers.Concatenate(name="final_output")([classification_output, regression_output])

    model = models.Model(inputs, outputs)
    return model


def build_model_pre1():
    """使用预训练的MobileNetV2"""
    # os.environ['http_proxy'] = 'http://127.0.0.1:10908'
    # os.environ['https_proxy'] = 'http://127.0.0.1:10908'
    # https://blog.csdn.net/weixin_44519481/article/details/110006997
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # 冻结预训练层
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Concatenate()([inputs, inputs, inputs])  # 灰度图复制为3通道
    # x = layers.Conv2D(3, (1, 1), activation='relu')(inputs)  # 1通道转3通道
    # x = base_model.output
    x = base_model(x)

    # 重新实现 用更小的网络
    # 指数衰减
    # dataset 8000 Epoch 120+20*4 到达 loss 0.8还是0.5 val_loss 0.9
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    global_features = layers.GlobalAveragePooling2D()(x)

    # 是否有数独的分类头
    classification = layers.Dense(128, activation='relu')(global_features)
    classification = layers.Dropout(0.3)(classification)
    classification = layers.Dense(64, activation='relu')(classification)
    classification_output = layers.Dense(1, activation='sigmoid', name='has_sudoku')(classification)
    
    # 关键点回归头 (16个点 * 2坐标 = 32个值)
    regression = layers.Dense(2048, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)
    regression = layers.Dense(1024, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)
    regression = layers.Dense(1024, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)
    regression = layers.Dense(256, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)
    regression = layers.Dense(256, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)
    regression = layers.Dense(256, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)
    regression = layers.Dense(256, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)
    regression = layers.Dense(256, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)
    regression = layers.Dense(128, activation='relu')(regression)
    regression = layers.Dense(64, activation='relu')(regression)
    regression_output = layers.Dense(32, activation='sigmoid', name='keypoints')(regression)  # 归一化坐标
    
    # 拼接成一个整体输出 (方便自定义损失)
    outputs = layers.Concatenate(name="final_output")([classification_output, regression_output])

    model = models.Model(inputs, outputs)
    return model


def build_model_pre2():
    """使用预训练的MobileNetV2"""
    # loss: 11.3172 - val_loss: 19.4054


    # os.environ['http_proxy'] = 'http://127.0.0.1:10908'
    # os.environ['https_proxy'] = 'http://127.0.0.1:10908'
    # https://blog.csdn.net/weixin_44519481/article/details/110006997
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # 冻结预训练层
    
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = layers.Concatenate()([inputs, inputs, inputs])  # 灰度图复制为3通道
    # x = layers.Conv2D(3, (1, 1), activation='relu')(inputs)  # 1通道转3通道
    # x = base_model.output
    x = base_model(x)

    # 叠加旧结构
    # outputs,has_sudoku,keypoints = build_model(x)
    # model = models.Model(inputs, outputs)
    # return model

    # 重新实现 用更小的网络
    # 公节点共头
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.Dense(512, activation='relu')(x)
    global_features = layers.GlobalAveragePooling2D()(x)

    # 是否有数独的分类头
    classification = layers.Dense(128, activation='relu')(global_features)
    classification = layers.Dropout(0.3)(classification)
    classification = layers.Dense(64, activation='relu')(classification)
    classification_output = layers.Dense(1, activation='sigmoid', name='has_sudoku')(classification)
    
    # 关键点回归头 (16个点 * 2坐标 = 32个值)
    regression = layers.Dense(512, activation='relu')(global_features)
    regression = layers.Dropout(0.3)(regression)

    # loss: 0.4212 val_loss: 1.0931
    # regression = layers.Dense(256, activation='relu')(regression)
    # regression = layers.Dropout(0.3)(regression)
    # regression = layers.Dense(256, activation='relu')(regression)
    # regression_output = layers.Dense(32, activation='sigmoid', name='keypoints')(regression)  # 归一化坐标
    # 拼接成一个整体输出 (方便自定义损失)
    # outputs = layers.Concatenate(name="final_output")([classification_output, regression_output])
    

    def point_net(reg,i):
        reg = layers.Dense(256, activation='relu')(reg)
        reg = layers.Dropout(0.3)(reg)
        reg = layers.Dense(256, activation='relu')(reg)
        reg = layers.Dense(2, activation='sigmoid', name=f'keypoints_{i}')(reg)  # 归一化坐标
        return reg
    
    regression_output = [point_net(regression,i) for i in range(16)]
    # 拼接成一个整体输出 (方便自定义损失)
    outputs = layers.Concatenate(name="final_output")([classification_output]+regression_output)


    model = models.Model(inputs, outputs)
    return model