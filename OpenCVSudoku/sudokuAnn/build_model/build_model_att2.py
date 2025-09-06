import tensorflow as tf
import numpy as np
from .const import IMG_SIZE

# ==================== 1. 轻量级注意力机制 ====================
class LightSpatialAttention(tf.keras.layers.Layer):
    """轻量级空间注意力机制"""
    def __init__(self):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(1, 3, padding='same', activation='sigmoid')
        
    def call(self, inputs):
        # 简化版：只用平均池化
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        attention = self.conv(avg_pool)
        return inputs * attention

class LightChannelAttention(tf.keras.layers.Layer):
    """轻量级通道注意力机制"""
    def __init__(self):
        super().__init__()
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc = tf.keras.layers.Dense(channels, activation='sigmoid')
        
    def call(self, inputs):
        attention = self.fc(self.gap(inputs))
        return inputs * attention[:, None, None, :]

# ==================== 2. 轻量级多尺度特征提取 ====================
class LightMultiScaleExtractor(tf.keras.layers.Layer):
    """轻量级多尺度特征提取器"""
    def __init__(self, filters):
        super().__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(filters//2, 1, padding='same')
        self.conv3x3 = tf.keras.layers.Conv2D(filters//2, 3, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        
    def call(self, inputs):
        branch1 = self.conv1x1(inputs)  # 细节特征
        branch2 = self.conv3x3(inputs)  # 局部特征
        
        combined = tf.concat([branch1, branch2], axis=-1)
        return self.activation(self.bn(combined))

# ==================== 3. 轻量级网络架构 ====================
def build_light_texture_locator():
    """构建轻量级纹理增强定位网络"""
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # 初始特征提取
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = LightSpatialAttention()(x)  # 空间注意力
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # 多尺度纹理特征提取
    x = LightMultiScaleExtractor(64)(x)
    x = LightChannelAttention()(x)  # 通道注意力
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # 中级特征提取
    x = LightMultiScaleExtractor(128)(x)
    x = LightSpatialAttention()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # 高级特征提取
    x = LightMultiScaleExtractor(256)(x)
    x = LightChannelAttention()(x)
    
    # 全局特征 - 使用全局最大池化减少参数量
    global_features = tf.keras.layers.GlobalMaxPooling2D()(x)
    global_features = tf.keras.layers.Dense(128, activation='relu')(global_features)
    global_features = tf.keras.layers.Dropout(0.3)(global_features)
    
    # 布尔值输出分支
    exists_output = tf.keras.layers.Dense(1, activation='sigmoid', name='exists')(global_features)
    
    # 坐标点输出分支
    coords_output = tf.keras.layers.Dense(32, activation='linear', name='coordinates')(global_features)
    coords_output = tf.keras.layers.Reshape((16, 2), name='coordinates_reshaped')(coords_output)
    
    model = tf.keras.Model(inputs, [exists_output, coords_output])
    
    # 打印模型大小
    # model.summary()
    print(f"模型参数数量: {model.count_params():,}")
    
    return model

# ==================== 4. 更小的网络版本 ====================
def build_mini_texture_locator():
    """构建迷你版纹理增强定位网络"""
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # 特征提取
    x = tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = LightSpatialAttention()(x)  # 保留空间注意力强化纹理
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = LightChannelAttention()(x)  # 保留通道注意力
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # 全局特征
    global_features = tf.keras.layers.GlobalAveragePooling2D()(x)
    global_features = tf.keras.layers.Dense(64, activation='relu')(global_features)
    global_features = tf.keras.layers.Dropout(0.2)(global_features)
    
    # 输出分支
    exists_output = tf.keras.layers.Dense(1, activation='sigmoid', name='exists')(global_features)
    coords_output = tf.keras.layers.Dense(32, activation='linear', name='coordinates')(global_features)
    coords_output = tf.keras.layers.Reshape((16, 2), name='coordinates_reshaped')(coords_output)
    
    model = tf.keras.Model(inputs, [exists_output, coords_output])
    
    # model.summary()
    print(f"迷你模型参数数量: {model.count_params():,}")
    
    return model

# ==================== 5. 带跳跃连接的轻量网络 ====================
def build_light_residual_locator():
    """带跳跃连接的轻量级网络"""
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    
    # 初始层
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # 残差块1
    shortcut = x
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    x = LightSpatialAttention()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # 残差块2
    shortcut = x
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # 如果需要调整shortcut的通道数
    shortcut = tf.keras.layers.Conv2D(64, 1, padding='same')(shortcut)
    shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.ReLU()(x)
    x = LightChannelAttention()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # 输出层
    global_features = tf.keras.layers.GlobalAveragePooling2D()(x)
    global_features = tf.keras.layers.Dense(128, activation='relu')(global_features)
    
    exists_output = tf.keras.layers.Dense(1, activation='sigmoid', name='exists')(global_features)
    coords_output = tf.keras.layers.Dense(32, activation='linear', name='coordinates')(global_features)
    coords_output = tf.keras.layers.Reshape((16, 2), name='coordinates_reshaped')(coords_output)
    
    model = tf.keras.Model(inputs, [exists_output, coords_output])
    
    # model.summary()
    print(f"残差模型参数数量: {model.count_params():,}")
    
    return model

# 使用示例
if __name__ == "__main__":
    # 选择其中一个模型
    model = build_light_texture_locator()      # 轻量版
    # model = build_mini_texture_locator()      # 迷你版  
    # model = build_light_residual_locator()    # 残差版

# 原始网络：约 2-3M 参数
# 轻量版：约 0.5-1M 参数
# 迷你版：约 0.1-0.3M 参数
# 残差版：约 0.3-0.8M 参数