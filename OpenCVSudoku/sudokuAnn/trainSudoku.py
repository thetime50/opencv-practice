import os
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers,regularizers

print('启动')
SATASET_FILE = os.path.join(os.path.dirname(__file__), 'dataset')
SATASET_FILE_IMG = os.path.join(SATASET_FILE, 'img')
SATASET_FILE_NPY = os.path.join(SATASET_FILE, 'sudoku_dataset.npy')
MODEL_TEMP_FILE = os.path.join(SATASET_FILE, 'sudoku_temp.h5')
MODEL_TEMP1_FILE = os.path.join(SATASET_FILE, 'sudoku_temp_1.h5')
MODEL_FILE = os.path.join(SATASET_FILE, 'sudoku.h5')

IMG_SIZE = 384  # 缩放到统一大小以保证 batch 内一致
BATCH_SIZE = 16

# --------------------------
# 自定义损失：条件关键点损失
# --------------------------

class ConditionalKeypointLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        """
        y_true: [batch, 33] -> 前1列是 has_sudoku, 后32列是关键点
        y_pred: [batch, 33] -> 同样结构
        """
        has_sudoku_true = y_true[:, 0]
        keypoints_true = y_true[:, 1:]
        has_sudoku_pred = y_pred[:, 0]
        keypoints_pred = y_pred[:, 1:]

        # 分类损失（二分类交叉熵）
        cls_loss = tf.keras.losses.binary_crossentropy(has_sudoku_true, has_sudoku_pred)

        # 关键点回归损失（只在 has_sudoku=1 时计算）
        mask = tf.expand_dims(has_sudoku_true, axis=-1)  # shape [batch,1]
        reg_loss = tf.reduce_mean(tf.square((keypoints_true - keypoints_pred) * mask), axis=-1)

        return cls_loss + reg_loss*1000

class ConditionalKeypointLoss2(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        """
        y_true: [batch, 33] -> 第0列是 has_sudoku, 后32列是关键点
        y_pred: [batch, 33] -> 同样结构
        """
        has_sudoku_true = y_true[:, 0]          # shape [B]
        keypoints_true = y_true[:, 1:]          # shape [B, 32]

        has_sudoku_pred = y_pred[:, 0]          # shape [B]
        keypoints_pred = y_pred[:, 1:]          # shape [B, 32]

        # -------------------
        # 分类损失（二分类交叉熵）
        # -------------------
        cls_loss = tf.keras.losses.binary_crossentropy(
            has_sudoku_true, has_sudoku_pred
        )

        # -------------------
        # 关键点回归损失
        # 只在 has_sudoku = 1 的样本上计算
        # -------------------
        mask = tf.cast(has_sudoku_true > 0.5, tf.float32)  # [B]
        mask_exp = tf.expand_dims(mask, axis=-1)           # [B,1]

        squared_error = tf.square(keypoints_true - keypoints_pred)  # [B,32]
        mse_per_sample = tf.reduce_mean(squared_error, axis=-1)     # [B]

        # 仅选取 has_sudoku=1 的样本
        reg_loss = tf.reduce_sum(mse_per_sample * mask) / (
            tf.reduce_sum(mask) + 1e-8
        )

        # -------------------
        # 合并
        # -------------------
        return cls_loss + reg_loss

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

# --------------------------
# 示例训练数据
# --------------------------
import numpy as np

# 加载数据
train_set, test_set = np.load(SATASET_FILE_NPY, allow_pickle=True)

train_images, train_has, train_points = train_set
test_images, test_has, test_points = test_set

# slice_repeat = lambda x,n : np.repeat(x[:n] , int(len(x)/n), axis=0)

# train_images = slice_repeat(train_images,100)
# train_has = slice_repeat(train_has,100)
# train_points = slice_repeat(train_points,100)

print("Train:", len(train_images), "Test:", len(test_images))

# # 数据处理函数
def parse_fn(img_path, has_sudoku, points):
    # 读取图片 回调里会变为张量
    img = tf.io.read_file(tf.strings.join([SATASET_FILE_IMG, img_path], separator=os.sep))
    img = tf.image.decode_jpeg(img, channels=1)  # 灰度图
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

    # y_true = [has_sudoku, 32个点]
    has_sudoku = tf.cast(has_sudoku, tf.float32)
    # points = tf.reshape(points, [-1])  # (32,)
    points = tf.cast(points, tf.float32)
    points = tf.reshape(points, [-1]) / IMG_SIZE
    y = tf.concat([[has_sudoku], points], axis=0)  # (33,)
    
    return img, y


# 训练集
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_has, train_points))
train_ds = train_ds.shuffle(10000).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 测试集
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_has, test_points))
test_ds = test_ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# --------------------------
# 模型编译
# --------------------------
# model = build_model()
# model = build_model_d()
model = build_model_pre()

# 固定学习率
# optimizer=tf.keras.optimizers.Adam(1e-4)
# 动态学习率 指数衰减
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-5, decay_steps=1000, decay_rate=0.82)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# 或者使用余弦退火
# lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
#     initial_learning_rate=1e-3, decay_steps=1000
# )
# optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=ConditionalKeypointLoss(),
    metrics=["accuracy"]
)
# model.summary()


# 加载模型和优化器状态
if os.path.exists(MODEL_TEMP_FILE):
    print("加载上次中断的模型")
    model.load_weights(MODEL_TEMP_FILE)

# 训练
print('开始训练')
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    MODEL_TEMP_FILE,
    save_weights_only=False, # True,
    save_best_only=True
)
model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20,
    callbacks=[checkpoint_callback]
)
print("保存模型")
model.save(MODEL_FILE)

if os.path.exists(MODEL_TEMP1_FILE):
    os.remove(MODEL_TEMP1_FILE)
if os.path.exists(MODEL_TEMP_FILE):
    os.rename(MODEL_TEMP_FILE, MODEL_TEMP1_FILE)
print('结束')
# --------------------------
# 推理
# --------------------------
# pred = model.predict(X)
# has_sudoku_pred = pred[:, 0]  # 概率
# points_pred = pred[:, 1:]     # 32个数
# print("Has sudoku prob:", has_sudoku_pred)
# print("Points shape:", points_pred.shape)


# 5000/5000 [===] - 439s 86ms/step - loss: 0.1872 - val_loss: 0.0804
# Epoch 2/20 5000/5000 [===] - 425s 85ms/step - loss: 0.0596 - val_loss: 0.0442
# Epoch 3/20 5000/5000 [===] - 408s 82ms/step - loss: 0.0397 - val_loss: 0.0313
# Epoch 4/20 5000/5000 [===] - 407s 81ms/step - loss: 0.0305 - val_loss: 0.0263
# Epoch 5/20 5000/5000 [===] - 406s 81ms/step - loss: 0.0246 - val_loss: 0.0207
# Epoch 6/20 5000/5000 [===] - 404s 81ms/step - loss: 0.0207 - val_loss: 0.0194
# Epoch 7/20 5000/5000 [===] - 403s 81ms/step - loss: 0.0182 - val_loss: 0.0170
# Epoch 8/20 5000/5000 [===] - 401s 80ms/step - loss: 0.0165 - val_loss: 0.0156
# Epoch 9/20 5000/5000 [===] - 401s 80ms/step - loss: 0.0150 - val_loss: 0.0145
# Epoch 10/20 5000/5000 [===] - 401s 80ms/step - loss: 0.0141 - val_loss: 0.0155
# Epoch 11/20 5000/5000 [===] - 400s 80ms/step - loss: 0.0130 - val_loss: 0.0142
# Epoch 12/20 5000/5000 [===] - 400s 80ms/step - loss: 0.0122 - val_loss: 0.0122
# Epoch 13/20 5000/5000 [===] - 402s 80ms/step - loss: 0.0114 - val_loss: 0.0107
# Epoch 14/20 5000/5000 [===] - 399s 80ms/step - loss: 0.0108 - val_loss: 0.0103
# Epoch 15/20 5000/5000 [===] - 396s 79ms/step - loss: 0.0102 - val_loss: 0.0120
# Epoch 16/20 5000/5000 [===] - 395s 79ms/step - loss: 0.0097 - val_loss: 0.0096
# Epoch 17/20 5000/5000 [===] - 396s 79ms/step - loss: 0.0090 - val_loss: 0.0107
# Epoch 18/20 5000/5000 [===] - 396s 79ms/step - loss: 0.0088 - val_loss: 0.0086
# Epoch 19/20 5000/5000 [===] - 396s 79ms/step - loss: 0.0083 - val_loss: 0.0092
# Epoch 20/20 5000/5000 [===] - 395s 79ms/step - loss: 0.0082 - val_loss: 0.0083
# 保存模型
# 结束