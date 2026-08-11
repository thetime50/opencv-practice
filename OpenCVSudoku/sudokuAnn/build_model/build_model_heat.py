import tensorflow as tf
from tensorflow.keras import layers, models

NUM_KEYPOINTS = 16
HEATMAP_STRIDE = 8


class SoftArgMax2D(layers.Layer):
    """对 (B,H,W,K) 热力图做 soft-argmax，输出归一化坐标 (B,K,2)，范围约 [0,1]。"""

    def __init__(self, beta=20.0, **kwargs):
        super().__init__(**kwargs)
        self.beta = beta

    def call(self, heatmaps):
        # heatmaps: logits 或概率图 (B,H,W,K)
        h = tf.shape(heatmaps)[1]
        w = tf.shape(heatmaps)[2]
        k = tf.shape(heatmaps)[3]

        flat = tf.reshape(heatmaps, (-1, h * w, k))  # (B, HW, K)
        weights = tf.nn.softmax(flat * self.beta, axis=1)  # 空间维 softmax

        y_coords = tf.cast(tf.range(h), tf.float32) / tf.cast(tf.maximum(h - 1, 1), tf.float32)
        x_coords = tf.cast(tf.range(w), tf.float32) / tf.cast(tf.maximum(w - 1, 1), tf.float32)
        yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')  # (H,W)
        xx = tf.reshape(xx, (1, h * w, 1))
        yy = tf.reshape(yy, (1, h * w, 1))

        x = tf.reduce_sum(weights * xx, axis=1)  # (B,K)
        y = tf.reduce_sum(weights * yy, axis=1)
        return tf.stack([x, y], axis=-1)  # (B,K,2)

    def get_config(self):
        config = super().get_config()
        config.update({'beta': self.beta})
        return config


def _residual_block(filters, shortcut, pooling=True):
    x = layers.Conv2D(filters, 3, padding='same', activation=None)(shortcut)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', activation=None)(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    if pooling:
        x = layers.MaxPooling2D(2)(x)
    return x


def build_heatmap_model(num_keypoints=NUM_KEYPOINTS, beta=20.0):
    """
    全卷积热力图模型，输入任意 H×W。
    输出:
      - has_sudoku: (B,1)
      - heatmaps:   (B,H/8,W/8,K)  sigmoid 热力图
      - keypoints:  (B,K,2)        soft-argmax 归一化坐标
    """
    inputs = layers.Input(shape=(None, None, 1), name='image')

    x = _residual_block(32, inputs, pooling=True)   # /2
    x = _residual_block(64, x, pooling=True)        # /4
    x = _residual_block(128, x, pooling=True)       # /8
    x = _residual_block(128, x, pooling=False)

    # 是否有数独
    gap = layers.GlobalAveragePooling2D()(x)
    shared = layers.Dense(128, activation='relu')(gap)
    shared = layers.Dropout(0.3)(shared)
    has_sudoku = layers.Dense(1, activation='sigmoid', name='has_sudoku')(shared)

    # 热力图头
    feat = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    heatmap_logits = layers.Conv2D(num_keypoints, 1, padding='same', name='heatmap_logits')(feat)
    heatmaps = layers.Activation('sigmoid', name='heatmaps')(heatmap_logits)

    # soft-argmax 得到坐标（作用在 logits 上更稳）
    keypoints = SoftArgMax2D(beta=beta, name='keypoints')(heatmap_logits)

    model = models.Model(
        inputs,
        {'has_sudoku': has_sudoku, 'heatmaps': heatmaps, 'keypoints': keypoints},
        name='sudoku_heatmap'
    )
    return model


tf.keras.utils.get_custom_objects().update({'SoftArgMax2D': SoftArgMax2D})
