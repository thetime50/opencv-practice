import tensorflow as tf
from tensorflow.keras import layers, models
from .const import IMG_SIZE
from .build_model_heat import SoftArgMax2D, NUM_KEYPOINTS, HEATMAP_STRIDE

# ---------------------------------------------------------------------------
# 可选配置：效果不好时按注释逐步切换 / 打开
# ---------------------------------------------------------------------------
# 
# | 特征层 | 用到层数 | 输出形状 | 约参数 | 步长 |
# |--------|----------|----------|--------|------|
# | `block_3_expand_relu` | 30 / 154 | **96×96×144** | 2.1万 | /4 |
# | `block_6_expand_relu` | 57 / 154 | **48×48×192** | 6.6万 | /8 |
# | `block_13_expand_relu` | 119 / 154 | **24×24×576** | 61.6万 | /16 |
# | `out_relu` | 154 / 154 | **12×12×1280** | 225.8万 | /32 |
# 
# 
# 1) 取哪一层特征（越浅分辨率越高；需配合上采样使热力图约为输#入的 1/HEATMAP_STRIDE）
# FEAT_LAYER = 'block_3_expand_relu'   # ~H/4  → 需 Downsample# 或改 HEATMAP_STRIDE=4
FEAT_LAYER = 'block_6_expand_relu'     # ~H/8  → 与 HEATMAP_STRIDE=8 对齐（推荐起步）
# FEAT_LAYER = 'block_13_expand_relu'  # ~H/16 → 需 Upsample×2
# FEAT_LAYER = 'out_relu'              # ~H/32 → 需 Upsample×4

# 2) 相对 FEAT_LAYER 的空间缩放（2=上采样, 1=不变, 0.5=下采样）
FEAT_SCALE = {
        'block_3_expand_relu': 0.5,# 配合 block_3（使输出仍为 /8）
        'block_6_expand_relu': 1, # 配合 block_6（使输出仍为 /8）
        'block_13_expand_relu': 2,# 配合 block_13 / out_relu
        'out_relu': 4,# 配合 out_relu（使输出仍为 /8）
    }[FEAT_LAYER]

# 3) 热力图头宽度（可训练参数主要在这里）
HEAD_FILTERS = 64
# HEAD_FILTERS = 128
# HEAD_FILTERS = 256

# 4) 分类头宽度
CLS_UNITS = 64
# CLS_UNITS = 128


# UNFREEZE_MODE = 'none'# 5) 解冻策略（默认全冻结；效果不够再改 UNFREEZE_MODE）
# UNFREEZE_MODE = 'none'       # 'none' | 'last_n' | 'from_block' | 'all'
UNFREEZE_MODE = 'last_n'   # 解冻最后 N 层
# UNFREEZE_MODE = 'from_block'  # 从某 block 起解冻
# UNFREEZE_MODE = 'all'
UNFREEZE_LAST_N = 20         # UNFREEZE_MODE == 'last_n'
UNFREEZE_FROM = 'block_6'   # UNFREEZE_MODE == 'from_block'


def _apply_freeze(base_model):
    """按 UNFREEZE_* 配置冻结 / 解冻 backbone。"""
    if UNFREEZE_MODE == 'none':
        base_model.trainable = False
    elif UNFREEZE_MODE == 'all':
        base_model.trainable = True
    elif UNFREEZE_MODE == 'last_n':
        base_model.trainable = True
        for layer in base_model.layers[:-UNFREEZE_LAST_N]:
            layer.trainable = False
    elif UNFREEZE_MODE == 'from_block':
        base_model.trainable = True
        trainable = False
        for layer in base_model.layers:
            if layer.name.startswith(UNFREEZE_FROM):
                trainable = True
            layer.trainable = trainable
    else:
        raise ValueError(f'不支持的 UNFREEZE_MODE={UNFREEZE_MODE}')

    return base_model


def _scale_features(x, scale):
    if scale == 1:
        return x
    if scale == 2:
        return layers.UpSampling2D(2, interpolation='bilinear')(x)
        # return layers.Conv2DTranspose(int(x.shape[-1] or HEAD_FILTERS), 3, strides=2, padding='same')(x)
    if scale == 4:
        x = layers.UpSampling2D(2, interpolation='bilinear')(x)
        return layers.UpSampling2D(2, interpolation='bilinear')(x)
    if scale == 0.5:
        return layers.MaxPooling2D(2)(x)
        # return layers.Conv2D(int(x.shape[-1] or HEAD_FILTERS), 3, strides=2, padding='same')(x)
    raise ValueError(f'不支持的 FEAT_SCALE={scale}')


def build_heatmap_model_pre(num_keypoints=NUM_KEYPOINTS, beta=20.0):
    """
    ImageNet 预训练 MobileNetV2 + 热力图头（backbone 默认冻结）。
    输出与 build_heatmap_model 一致:
      has_sudoku (B,1), heatmaps (B,H/8,W/8,K), keypoints (B,K,2)
    输入建议为 IMG_SIZE 的倍数（MobileNet 常用约束），训练侧仍用 HEATMAP_STRIDE=8。
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
    )
    base_model = _apply_freeze(base_model)

    # 截取浅层/中层特征，避免默认 /32 过粗
    feat_extractor = models.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer(FEAT_LAYER).output,
        name='mobilenet_feat',
    )
    feat_extractor.trainable = base_model.trainable
    # 若 base 部分解冻，同步子模型
    for l_src, l_dst in zip(base_model.layers, feat_extractor.layers):
        l_dst.trainable = l_src.trainable

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='image')
    x = layers.Concatenate()([inputs, inputs, inputs])
    # x = layers.Conv2D(3, 1, padding='same', name='gray_to_rgb')(inputs)  # 可选可学习 1→3

    feat = feat_extractor(x)
    feat = _scale_features(feat, FEAT_SCALE)

    # ---- 分类头（小）----
    gap = layers.GlobalAveragePooling2D()(feat)
    shared = layers.Dense(CLS_UNITS, activation='relu')(gap)
    # shared = layers.Dense(CLS_UNITS * 2, activation='relu')(gap)  # 加大分类头
    shared = layers.Dropout(0.3)(shared)
    has_sudoku = layers.Dense(1, activation='sigmoid', name='has_sudoku')(shared)

    # ---- 热力图头（小）----
    h = layers.Conv2D(HEAD_FILTERS, 3, padding='same', activation='relu')(feat)
    # h = layers.Conv2D(HEAD_FILTERS, 3, padding='same', activation='relu')(h)  # 加一层
    # h = layers.Conv2D(HEAD_FILTERS, 3, padding='same', activation='relu')(h)
    heatmap_logits = layers.Conv2D(num_keypoints, 1, padding='same', name='heatmap_logits')(h)
    heatmaps = layers.Activation('sigmoid', name='heatmaps')(heatmap_logits)
    keypoints = SoftArgMax2D(beta=beta, name='keypoints')(heatmap_logits)

    model = models.Model(
        inputs,
        {'has_sudoku': has_sudoku, 'heatmaps': heatmaps, 'keypoints': keypoints},
        name='sudoku_heatmap_pre',
    )
    return model
