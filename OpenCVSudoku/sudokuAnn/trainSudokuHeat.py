import os
import tensorflow as tf
import numpy as np
from build_model import (
    BATCH_SIZE,
    IMG_SIZE,
    SoftArgMax2D,
    build_heatmap_model,
    build_heatmap_model_pre,
    HEATMAP_STRIDE,
    NUM_KEYPOINTS,
)
from const import (
    SATASET_FILE_IMG,
    SATASET_FILE_NPY,
    HEAT_MODEL_TEMP_FILE as MODEL_TEMP_FILE,
    HEAT_MODEL_TEMP1_FILE as MODEL_TEMP1_FILE,
    HEAT_MODEL_FILE as MODEL_FILE,
)

print('启动')

HEATMAP_SIGMA = 2.0
KEYPOINTS_LOSS_WEIGHT = 50.0  # 加大坐标权重（原 10）
NEAR_PEAK_RATIO = 0.25        # 峰附近：|dx|<W/4 且 |dy|<H/4


def pad_to_multiple(img, stride=HEATMAP_STRIDE):
    """将 H、W pad 到 stride 的倍数（右下填充），支持任意尺寸。"""
    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    nh = ((h + stride - 1) // stride) * stride
    nw = ((w + stride - 1) // stride) * stride
    return tf.image.pad_to_bounding_box(img, 0, 0, nh, nw)


def make_gaussian_heatmaps(points_norm, hm_h, hm_w, has_sudoku, sigma=HEATMAP_SIGMA):
    """
    points_norm: (K,2) 相对整图归一化坐标 [0,1]，顺序 (x,y)
    返回: (hm_h, hm_w, K)
    """
    has = tf.cast(has_sudoku > 0.5, tf.float32)
    hm_h = tf.cast(hm_h, tf.int32)
    hm_w = tf.cast(hm_w, tf.int32)

    y_coords = tf.cast(tf.range(hm_h), tf.float32)
    x_coords = tf.cast(tf.range(hm_w), tf.float32)
    yy, xx = tf.meshgrid(y_coords, x_coords, indexing='ij')  # (H,W)
    xx = xx[:, :, tf.newaxis]  # (H,W,1)
    yy = yy[:, :, tf.newaxis]

    # 映射到热力图像素坐标
    gx = points_norm[:, 0] * tf.cast(tf.maximum(hm_w - 1, 1), tf.float32)  # (K,)
    gy = points_norm[:, 1] * tf.cast(tf.maximum(hm_h - 1, 1), tf.float32)

    dx = xx - gx[tf.newaxis, tf.newaxis, :]
    dy = yy - gy[tf.newaxis, tf.newaxis, :]
    heat = tf.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    heat = heat * has
    return heat


class NearPeakHeatmapMSE(tf.keras.losses.Loss):
    """仅在 GT 峰附近（|dx|<W/4 且 |dy|<H/4）计算热力图 MSE。"""

    def __init__(self, ratio=NEAR_PEAK_RATIO, name='near_peak_heatmap_mse'):
        super().__init__(name=name)
        self.ratio = ratio

    def call(self, y_true, y_pred):
        # (B,H,W,K)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        h = tf.shape(y_true)[1]
        w = tf.shape(y_true)[2]
        k = tf.shape(y_true)[3]
        hf = tf.cast(h, tf.float32)
        wf = tf.cast(w, tf.float32)

        flat = tf.reshape(y_true, (-1, h * w, k))
        peak_idx = tf.argmax(flat, axis=1, output_type=tf.int32)  # (B,K)
        py = tf.cast(peak_idx // w, tf.float32)
        px = tf.cast(peak_idx % w, tf.float32)

        yy = tf.cast(tf.range(h), tf.float32)[:, tf.newaxis]  # (H,1)
        xx = tf.cast(tf.range(w), tf.float32)[tf.newaxis, :]  # (1,W)
        dx = tf.abs(xx[tf.newaxis, :, :, tf.newaxis] - px[:, tf.newaxis, tf.newaxis, :])
        dy = tf.abs(yy[tf.newaxis, :, :, tf.newaxis] - py[:, tf.newaxis, tf.newaxis, :])
        mask = tf.cast(
            (dx < wf * self.ratio) & (dy < hf * self.ratio),
            tf.float32,
        )

        sq = tf.square(y_true - y_pred) * mask
        denom = tf.reduce_sum(mask, axis=[1, 2, 3]) + 1e-6
        return tf.reduce_sum(sq, axis=[1, 2, 3]) / denom


class NearPeakKeypointMSE(tf.keras.losses.Loss):
    """
    坐标损失：以 GT 为峰，仅当 |dx|<ratio 且 |dy|<ratio（相对整图宽高）时计入该点；
    远点用 far_weight，避免训练初期梯度全无。
    """

    def __init__(self, ratio=NEAR_PEAK_RATIO, far_weight=0.1, name='near_peak_keypoint_mse'):
        super().__init__(name=name)
        self.ratio = ratio
        self.far_weight = far_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        dx = tf.abs(y_true[..., 0] - y_pred[..., 0])
        dy = tf.abs(y_true[..., 1] - y_pred[..., 1])
        near = tf.cast((dx < self.ratio) & (dy < self.ratio), tf.float32)
        w = near * 1.0 + (1.0 - near) * self.far_weight
        sq = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
        return tf.reduce_mean(sq * w, axis=-1)


class SudokuHeatTrainer:
    def __init__(self, model):
        self.model = model
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3, decay_steps=1000, decay_rate=0.88)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.train_ds = None
        self.test_ds = None
        self.dataset_init()
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_TEMP_FILE,
                save_weights_only=False,
                save_best_only=True,
                monitor='val_loss',
            )
        ]

    def dataset_init(self, dataset_path=SATASET_FILE_NPY):
        slice_dataset = lambda ds, lens: [_[:lens] for _ in ds]

        train_set, test_set = np.load(dataset_path, allow_pickle=True)
        train_images, train_has, train_points = train_set
        test_images, test_has, test_points = slice_dataset(test_set, 5000)

        print('Train:', len(train_images), 'Test:', len(test_images))

        # slice_repeat = lambda x,n : np.repeat(x[:n] , int(len(x)/n), axis=0)

        # train_images = slice_repeat(train_images,100)
        # train_has = slice_repeat(train_has,100)
        # train_points = slice_repeat(train_points,100)
        def parse_fn(img_path, has_sudoku, points):
            img = tf.io.read_file(tf.strings.join([SATASET_FILE_IMG, img_path], separator=os.sep))
            img = tf.io.decode_image(img, channels=1, expand_animations=False)
            img.set_shape([None, None, 1])
            img = tf.image.convert_image_dtype(img, tf.float32)

            # 可选：统一缩放到 IMG_SIZE（保持可训练 batch）；网络本身仍支持任意尺寸
            # 若要纯原图训练，注释掉下一行即可（需配合 padded_batch）
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
            img = pad_to_multiple(img, HEATMAP_STRIDE)

            h = tf.shape(img)[0]
            w = tf.shape(img)[1]
            hm_h = h // HEATMAP_STRIDE
            hm_w = w // HEATMAP_STRIDE

            has_sudoku = tf.cast(has_sudoku, tf.float32)
            points = tf.cast(tf.reshape(points, (NUM_KEYPOINTS, 2)), tf.float32)
            # 数据集坐标为生成图像素；resize 到 IMG_SIZE 后按 IMG_SIZE 归一化
            # 若注释掉上方 resize、改用原图，请改为: points / [w,h]（pad 前尺寸）
            points_norm = points / float(IMG_SIZE)

            heatmaps = make_gaussian_heatmaps(
                points_norm, hm_h, hm_w, has_sudoku, sigma=HEATMAP_SIGMA)

            y = {
                'has_sudoku': tf.reshape(has_sudoku, (1,)),
                'heatmaps': heatmaps,
                'keypoints': points_norm,
            }
            # has_sudoku=0 时屏蔽热力图与坐标损失
            # 自定义 loss 已 reduce 到 (B,)，sample_weight 用每样本标量
            sw = {
                'has_sudoku': tf.constant(1.0),
                # 'heatmaps': tf.reshape(has_sudoku, (1, 1)),
                # 'keypoints': tf.reshape(has_sudoku, (1,)),
                'heatmaps': has_sudoku,
                'keypoints': has_sudoku,
            }
            return img, y, sw

        def make_ds(images, has, points, shuffle):
            ds = tf.data.Dataset.from_tensor_slices((images, has, points))
            if shuffle:
                ds = ds.shuffle(10000)
            ds = ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
            # 任意尺寸：同 batch pad；固定 IMG_SIZE 时等价于普通 batch
            ds = ds.padded_batch(
                BATCH_SIZE,
                padded_shapes=(
                    [None, None, 1],
                    {
                        'has_sudoku': [1],
                        'heatmaps': [None, None, NUM_KEYPOINTS],
                        'keypoints': [NUM_KEYPOINTS, 2],
                    },
                    {
                        'has_sudoku': [],
                        # 'heatmaps': [1, 1],
                        # 'keypoints': [1],
                        'heatmaps': [],
                        'keypoints': [],
                    },
                ),
                padding_values=(
                    0.0,
                    {
                        'has_sudoku': 0.0,
                        'heatmaps': 0.0,
                        'keypoints': 0.0,
                    },
                    {
                        'has_sudoku': 0.0,
                        'heatmaps': 0.0,
                        'keypoints': 0.0,
                    },
                ),
            )
            return ds.prefetch(tf.data.AUTOTUNE)

        self.train_ds = make_ds(train_images, train_has, train_points, shuffle=True)
        self.test_ds = make_ds(test_images, test_has, test_points, shuffle=False)

    def before_train(self):
        self.model.compile(
            optimizer=self.optimizer,
            loss={
                'has_sudoku': 'binary_crossentropy',
                # 'heatmaps': 'mse',
                # 'keypoints': 'mse',
                'heatmaps': NearPeakHeatmapMSE(),
                'keypoints': NearPeakKeypointMSE(),
            },
            loss_weights={
                'has_sudoku': 1.0,
                'heatmaps': 1.0,
                'keypoints': KEYPOINTS_LOSS_WEIGHT, # （原 10）
            },
            metrics={
                'has_sudoku': ['accuracy'],
                'keypoints': ['mae'],
            },
        )
        if os.path.exists(MODEL_TEMP_FILE):
            print('加载上次中断的模型')
            self.model.load_weights(MODEL_TEMP_FILE)

    def after_train(self):
        if os.path.exists(MODEL_TEMP1_FILE):
            os.remove(MODEL_TEMP1_FILE)
        if os.path.exists(MODEL_TEMP_FILE):
            os.rename(MODEL_TEMP_FILE, MODEL_TEMP1_FILE)
        print('结束')

    def train(self, cnt=5, epochs=20):
        self.before_train()
        print('开始训练')
        for _ in range(cnt):
            self.model.fit(
                self.train_ds,
                validation_data=self.test_ds,
                epochs=epochs,
                callbacks=self.callbacks,
            )
            print('保存模型')
            self.model.save(MODEL_FILE, custom_objects={'SoftArgMax2D': SoftArgMax2D})
        self.after_train()


if __name__ == '__main__':
    # model = build_heatmap_model()
    model = build_heatmap_model_pre()
    model.summary()
    trainable = sum(int(tf.keras.backend.count_params(w)) for w in model.trainable_weights)
    total = model.count_params()
    print(f'可训练参数: {trainable:,} / 总参数: {total:,}')
    trainer = SudokuHeatTrainer(model)
    trainer.train(1)
