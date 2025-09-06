import tensorflow as tf
import numpy as np
from .const import IMG_SIZE

# ==================== 1. 注意力机制 ====================
class SpatialAttention(tf.keras.layers.Layer):
    """空间注意力机制，增强纹理区域"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(1, kernel_size, 
                                         padding='same', 
                                         activation='sigmoid')
        
    def call(self, inputs):
        # 通道平均和最大池化
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return inputs * attention

class ChannelAttention(tf.keras.layers.Layer):
    """通道注意力机制，聚焦重要特征通道"""
    def __init__(self, ratio=8):
        super().__init__()
        self.ratio = ratio
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
        self.gmp = tf.keras.layers.GlobalMaxPooling2D()
        
    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = tf.keras.layers.Dense(channels // self.ratio, activation='relu')
        self.fc2 = tf.keras.layers.Dense(channels, activation='sigmoid')
        
    def call(self, inputs):
        avg_out = self.fc2(self.fc1(self.gap(inputs)))
        max_out = self.fc2(self.fc1(self.gmp(inputs)))
        attention = tf.sigmoid(avg_out + max_out)
        return inputs * attention[:, None, None, :]

# ==================== 2. 多尺度特征提取 ====================
class MultiScaleTextureExtractor(tf.keras.layers.Layer):
    """多尺度纹理特征提取器"""
    def __init__(self, filters):
        super().__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(filters, 1, padding='same')
        self.conv3x3 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.conv5x5 = tf.keras.layers.Conv2D(filters, 5, padding='same')
        self.conv_dilated = tf.keras.layers.Conv2D(filters, 3, 
                                                 padding='same', 
                                                 dilation_rate=2)
        self.bn = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.ReLU()
        
    def call(self, inputs):
        branch1 = self.conv1x1(inputs)  # 捕捉细节特征
        branch2 = self.conv3x3(inputs)  # 中等尺度特征
        branch3 = self.conv5x5(inputs)  # 大尺度特征
        branch4 = self.conv_dilated(inputs)  # 扩大感受野
        
        combined = tf.concat([branch1, branch2, branch3, branch4], axis=-1)
        return self.activation(self.bn(combined))

# ==================== 3. 网络架构 ====================
def build_texture_enhanced_locator():
    """构建纹理增强的多输出定位网络"""
    inputs = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1))
    x = tf.keras.layers.Concatenate()([inputs, inputs, inputs])  # 灰度图复制为3通道
    
    # 共享特征提取主干
    x = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = SpatialAttention()(x)  # 空间注意力
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # 多尺度纹理特征提取
    x = MultiScaleTextureExtractor(128)(x)
    x = ChannelAttention()(x)  # 通道注意力
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # 中级特征提取
    x = MultiScaleTextureExtractor(256)(x)
    x = SpatialAttention()(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    
    # 高级特征提取
    x = MultiScaleTextureExtractor(512)(x)
    x = ChannelAttention()(x)
    
    # 全局特征
    global_features = tf.keras.layers.GlobalAveragePooling2D()(x)
    global_features = tf.keras.layers.Dense(256, activation='relu')(global_features)
    global_features = tf.keras.layers.Dropout(0.3)(global_features)
    
    # 布尔值输出分支（是否存在）
    exists_output = tf.keras.layers.Dense(1, activation='sigmoid', name='exists')(global_features)
    
    # 坐标点输出分支（16个点，每个点x,y）
    coords_output = tf.keras.layers.Dense(32, activation='linear', name='coordinates')(global_features)  # 32 = 16*2
    coords_output = tf.keras.layers.Reshape((16, 2), name='coordinates_reshaped')(coords_output)
    
    model = tf.keras.Model(inputs, [exists_output, coords_output])
    return model
