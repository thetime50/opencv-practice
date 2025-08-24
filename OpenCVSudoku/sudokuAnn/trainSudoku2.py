import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os

'''
逻辑有问题不能运行
regression_output  ouyput是长度32 y_train['keypoints'] = y_train_combined 长度是33
'''

SATASET_FILE = os.path.join(os.path.dirname(__file__), 'dataset')
SATASET_FILE_IMG = os.path.join(SATASET_FILE, 'img')
SATASET_FILE_NPY = os.path.join(SATASET_FILE, 'sudoku_dataset.npy')
MODEL_TEMP_FILE = os.path.join(SATASET_FILE, 'd_sudoku_temp.h5')
MODEL_TEMP1_FILE = os.path.join(SATASET_FILE, 'd_sudoku_temp_1.h5')
MODEL_FILE = os.path.join(SATASET_FILE, 'd_sudoku.h5')


# 设置日志级别
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['OPENCV_LOG_LEVEL'] = 'OFF'

def create_sudoku_detection_model():
    """
    创建支持任意尺寸的数独检测模型
    输出: [是否有数独, 16个关键点坐标(x1,y1,x2,y2,...,x16,y16)]
    """
    inputs = keras.Input(shape=(None, None, 1))  # 任意尺寸输入
    
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
    
    model = keras.Model(inputs=inputs, outputs=[classification_output, regression_output])
    return model

def masked_keypoints_loss(y_true, y_pred):
    """
    自定义损失函数，当has_sudoku为False时屏蔽keypoints损失
    """
    # y_true格式: [has_sudoku, keypoints]
    has_sudoku = y_true[:, 0]  # 第一列是has_sudoku标签
    keypoints_true = y_true[:, 1:]  # 剩余是keypoints坐标
    has_sudoku_pred = y_pred[:, 0]  # 第一列是has_sudoku标签
    keypoints_pred = y_pred[:, 1:]  # 剩余是keypoints坐标
    
    # # 分类损失（二分类交叉熵）
    # cls_loss = tf.keras.losses.binary_crossentropy(has_sudoku, has_sudoku_pred)

    # 计算MSE损失
    mse_loss = tf.keras.losses.mean_squared_error(keypoints_true, keypoints_pred)
    
    # 当has_sudoku为False时，损失为0
    mask = tf.cast(has_sudoku, tf.float32)  # True=1, False=0
    masked_loss = mse_loss * mask
    
    return tf.reduce_mean(masked_loss) # cls_loss + 


def train_model(model, train_data, val_data=None, epochs=20, batch_size=16):
    """
    训练模型
    """
    # 编译模型
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'has_sudoku': 'binary_crossentropy',
            'keypoints': masked_keypoints_loss
        },
        loss_weights={
            'has_sudoku': 1.0,
            'keypoints': 1.0
        },
        metrics={
            'has_sudoku': ['accuracy'],# 'precision',  'recall',
            'keypoints': 'mae'
        }
    )
    
# 加载模型和优化器状态
    if os.path.exists(MODEL_TEMP_FILE):
        model.load_weights(MODEL_TEMP_FILE)
        print("加载上次中断的模型")

    # 回调函数
    callbacks = [
        keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10),
        keras.callbacks.ModelCheckpoint(MODEL_TEMP_FILE, save_best_only=True)
    ]
    
    # 训练
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=epochs,
        # batch_size=batch_size, # dataset 优先
        callbacks=callbacks,
        verbose=1
    )
    print("保存模型")
    model.save(MODEL_FILE)

    if os.path.exists(MODEL_TEMP1_FILE):
        os.remove(MODEL_TEMP1_FILE)
    if os.path.exists(MODEL_TEMP_FILE):
        os.rename(MODEL_TEMP_FILE, MODEL_TEMP1_FILE)
    print('结束')
    
    return history

# # 数据处理函数
IMG_SIZE = 384  # 缩放到统一大小以保证 batch 内一致
BATCH_SIZE = 16
def parse_fn(img_path, has_sudoku, points):
    # 读取图片 回调里会变为张量
    img = tf.io.read_file(tf.strings.join([SATASET_FILE_IMG, img_path], separator=os.sep))
    img = tf.image.decode_jpeg(img, channels=1)  # 灰度图
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]

    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))/ 255.0

    # y_true = [has_sudoku, 32个点]
    has_sudoku = tf.cast(has_sudoku, tf.float32)
    # points = tf.reshape(points, [-1])  # (32,)
    points = tf.cast(points, tf.float32)
    points = tf.reshape(points, [-1]) / IMG_SIZE
    y = tf.concat([[has_sudoku], points], axis=0)  # (33,)
    
    return img, {
            'has_sudoku': has_sudoku,
            'keypoints': y
        }
def load_training_data():
    train_set, test_set = np.load(SATASET_FILE_NPY, allow_pickle=True)
    # 训练集
    train_ds = tf.data.Dataset.from_tensor_slices((train_set[0],train_set[1],train_set[2])) # train_images, train_has, train_points
    train_ds = train_ds.shuffle(10000).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 测试集
    test_ds = tf.data.Dataset.from_tensor_slices((test_set[0],test_set[1],test_set[2]))
    test_ds = test_ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds,test_ds

# 使用示例
def main():
    
    # 训练模型（如果有数据）
    train_set, test_set = load_training_data()
    model=create_sudoku_detection_model()
    history = train_model(model, train_set, test_set,epochs=20,batch_size=BATCH_SIZE)    

if __name__ == "__main__":
    main()