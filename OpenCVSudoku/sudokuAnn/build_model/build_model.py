import os
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers,regularizers

# --------------------------
# 模型定义
# --------------------------
def build_model(inputs = None):
    return_model = False
    if inputs is None:
        return_model = True
        inputs = layers.Input(shape=(None, None, 1))  # 任意尺寸输入

    # Backbone 卷积
    # x = layers.Conv2D(32, 3, activation="relu", padding="same")(inputs)
    # x = layers.MaxPooling2D(2)(x)
    # x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    # x = layers.MaxPooling2D(2)(x)
    # x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    # x = layers.MaxPooling2D(2)(x)

    # 收敛更快 看看能不能突破无法收敛的问题
    def residualConnection(filters,shortcut,pooling = True):
        x = layers.Conv2D(filters, 3, padding="same", activation=None)(shortcut)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation=None)(x)
        x = layers.BatchNormalization()(x)
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding="same")(shortcut)
        x = layers.Add()([x, shortcut])  # 残差连接
        x = layers.ReLU()(x)
        if(pooling):
            x = layers.MaxPooling2D(2)(x)

        return x
    
    x = residualConnection(32,inputs)
    x = residualConnection(64,x)
    x = residualConnection(128,x)
    # x = residualConnection(128,x,False) # 加了一层 卡在loss: 5.4433 - val_loss: 5.2975

    # 全局池化
    x = layers.GlobalAveragePooling2D()(x)

    # 公共特征层
    shared = layers.Dense(256, activation="relu",)(x) # kernel_regularizer=regularizers.l2(0.01)
    shared = layers.Dropout(0.3)(shared)
    shared = layers.Dense(128, activation="relu",)(shared) # kernel_regularizer=regularizers.l2(0.01)

    # 输出1: 是否有数独
    has_sudoku = layers.Dense(1, activation="sigmoid", name="has_sudoku")(shared)

    # 输出2: 关键点 (32个数, 归一化坐标)
    keypoints = layers.Dense(256, activation="relu",)(shared) # kernel_regularizer=regularizers.l2(0.01)
    keypoints = layers.Dropout(0.3)(keypoints)
    keypoints = layers.Dense(32, activation="sigmoid", name="points")(keypoints)

    # 拼接成一个整体输出 (方便自定义损失)
    outputs = layers.Concatenate(name="final_output")([has_sudoku, keypoints])

    if(return_model):
        model = models.Model(inputs, outputs)
        return model
    else:
        return outputs,has_sudoku,keypoints





def build_model_d():
    """
    创建支持任意尺寸的数独检测模型
    输出: [是否有数独, 16个关键点坐标(x1,y1,x2,y2,...,x16,y16)]
    """
    inputs = layers.Input(shape=(None, None, 1))  # 任意尺寸输入
    
    # 特征提取 backbone (全卷积架构)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    
    # 全局特征提取
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
    
    # model = models.Model(inputs=inputs, outputs=[classification_output, regression_output])
    outputs = layers.Concatenate(name="final_output")([classification_output, regression_output])
    model = models.Model(inputs=inputs, outputs=outputs)
    return model