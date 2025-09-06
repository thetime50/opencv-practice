import tensorflow as tf
import numpy as np
import os
from build_model import BATCH_SIZE,IMG_SIZE, build_texture_enhanced_locator,\
    build_light_texture_locator,\
    build_mini_texture_locator,\
    build_light_residual_locator

from const import SATASET_FILE,\
    SATASET_FILE_IMG,\
    SATASET_FILE_NPY,\
    MODEL_TEMP_FILE_ATT as MODEL_TEMP_FILE,\
    MODEL_TEMP1_FILE_ATT as MODEL_TEMP1_FILE,\
    MODEL_FILE_ATT as MODEL_FILE

# ==================== 4. 损失函数 ====================
class MultiTaskLoss(tf.keras.losses.Loss):
    """多任务损失函数：布尔值分类 + 坐标回归"""
    def __init__(self, exist_weight=1.0, coord_weight=1.0, texture_weight=0.1):
        super().__init__()
        self.exist_weight = exist_weight
        self.coord_weight = coord_weight
        self.texture_weight = texture_weight
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        
    def call(self, y_true, y_pred):
        # y_true: [exists_true, coords_true]
        # y_pred: [exists_pred, coords_pred]
        exists_true = y_true[0]
        coords_true = y_true[1]
        exists_pred = y_pred[0]
        coords_pred = y_pred[1]

        
        # 布尔值分类损失
        exist_loss = self.bce_loss(exists_true, exists_pred)
        
        # 坐标回归损失（只对存在的样本计算）
        mask = tf.cast(exists_true, tf.float32)  # 存在掩码
        coord_loss = self.mse_loss(coords_true * mask[..., tf.newaxis], 
                                 coords_pred * mask[..., tf.newaxis])
        
        # 纹理梯度一致性损失
        texture_loss = self._texture_consistency_loss(coords_true, coords_pred, mask)
        
        total_loss = (self.exist_weight * exist_loss + 
                     self.coord_weight * coord_loss + 
                     self.texture_weight * texture_loss)
        
        return total_loss
    
    def _texture_consistency_loss(self, coords_true, coords_pred, mask):
        """纹理梯度一致性损失"""
        # 只对存在的样本计算
        valid_true = coords_true * mask[..., tf.newaxis]
        valid_pred = coords_pred * mask[..., tf.newaxis]
        
        # 计算坐标间的梯度（模拟纹理变化）
        true_grad = tf.abs(valid_true[:, 1:, :] - valid_true[:, :-1, :])
        pred_grad = tf.abs(valid_pred[:, 1:, :] - valid_pred[:, :-1, :])
        
        return tf.reduce_mean(tf.abs(true_grad - pred_grad))

class AdaptiveMultiTaskLoss(tf.keras.losses.Loss):
    """自适应多任务损失，动态调整权重"""
    def __init__(self):
        super().__init__()
        self.bce_loss = tf.keras.losses.BinaryCrossentropy()
        self.mse_loss = tf.keras.losses.MeanSquaredError()
        # 可学习的损失权重
        self.log_vars = tf.Variable([0.0, 0.0, 0.0], trainable=True)
        
    def call(self, y_true, y_pred):
        exists_true, coords_true = y_true
        exists_pred, coords_pred = y_pred
        
        # 计算各个损失
        exist_loss = self.bce_loss(exists_true, exists_pred)
        
        mask = tf.cast(exists_true, tf.float32)
        coord_loss = self.mse_loss(coords_true * mask[..., tf.newaxis], 
                                 coords_pred * mask[..., tf.newaxis])
        
        texture_loss = self._texture_consistency_loss(coords_true, coords_pred, mask)
        
        # 使用可学习权重
        precision1 = tf.exp(-self.log_vars[0])
        precision2 = tf.exp(-self.log_vars[1])
        precision3 = tf.exp(-self.log_vars[2])
        
        total_loss = (precision1 * exist_loss + precision2 * coord_loss + 
                     precision3 * texture_loss + self.log_vars[0] + 
                     self.log_vars[1] + self.log_vars[2])
        
        return total_loss
    
    def _texture_consistency_loss(self, coords_true, coords_pred, mask):
        """纹理梯度一致性损失"""
        valid_true = coords_true * mask[..., tf.newaxis]
        valid_pred = coords_pred * mask[..., tf.newaxis]
        
        true_grad = tf.abs(valid_true[:, 1:, :] - valid_true[:, :-1, :])
        pred_grad = tf.abs(valid_pred[:, 1:, :] - valid_pred[:, :-1, :])
        
        return tf.reduce_mean(tf.abs(true_grad - pred_grad))

# ==================== 5. 数据增强 ====================
class MultiOutputAugmentation:
    """多输出数据增强"""
    def __init__(self):
        self.augmentations = [
            self._add_texture_noise,
            self._adjust_texture_contrast,
            self._apply_flip,
            self._simulate_lighting_changes
        ]
    
    def _add_texture_noise(self, image):
        """添加纹理噪声"""
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.05)
        return image + noise
    
    def _adjust_texture_contrast(self, image):
        """调整纹理对比度"""
        return tf.image.random_contrast(image, 0.8, 1.2)
    
    def _apply_flip(self, image, coords):
        """水平翻转并调整坐标"""
        image = tf.image.flip_left_right(image)
        coords = coords.numpy() if hasattr(coords, 'numpy') else coords
        coords = coords.copy()
        coords[:, :, 0] = 1.0 - coords[:, :, 0]  # 假设坐标是归一化的[0,1]
        return image, coords
    
    def _simulate_lighting_changes(self, image):
        """模拟光照变化"""
        return tf.image.random_brightness(image, 0.2)
    
    def augment(self, image, exists_label, coords_label):
        """应用增强"""
        # 随机选择增强方法
        for aug_func in self.augmentations:
            if tf.random.uniform(()) > 0.7:
                if aug_func == self._apply_flip:
                    image, coords_label = self._apply_flip(image, coords_label)
                else:
                    image = aug_func(image)
        
        return image, exists_label, coords_label

# ==================== 6. 训练器 ====================
class MultiOutputTrainer:
    """多输出训练器"""
    def __init__(self, model, initial_lr=1e-3, use_adaptive_loss=False):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
        self.use_adaptive_loss = use_adaptive_loss
        
        if use_adaptive_loss:
            self.loss_fn = AdaptiveMultiTaskLoss()
        else:
            self.loss_fn = MultiTaskLoss()
        
        # 编译训练函数
        self.train_step = tf.function(
            self._train_step,
            input_signature=[
                tf.TensorSpec(shape=[None, None, None, 1], dtype=tf.float32),
                tf.TensorSpec(shape=[None, 1], dtype=tf.float32),  # exists
                tf.TensorSpec(shape=[None, 16, 2], dtype=tf.float32)  # coords
            ]
        )
        self.dataset_init()
    
    def dataset_init(self,dataset_path = SATASET_FILE_NPY):
        slice_dataset = lambda ds,lens:[_[:lens] for _ in ds]

        # 加载数据
        train_set, test_set = np.load(dataset_path, allow_pickle=True)

        train_images, train_has, train_points = train_set
        test_images, test_has, test_points = slice_dataset(test_set,5000)

        # slice_repeat = lambda x,n : np.repeat(x[:n] , int(len(x)/n), axis=0)

        # train_images = slice_repeat(train_images,100)
        # train_has = slice_repeat(train_has,100)
        # train_points = slice_repeat(train_points,100)
        def parse_fn(img_path, has_sudoku, points):
            # 读取图片
            img = tf.io.read_file(tf.strings.join([SATASET_FILE_IMG, img_path], separator=os.sep))
            img = tf.image.decode_jpeg(img, channels=1)  # 灰度图
            img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
            img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))

            # 修改为多输出格式：返回元组而不是拼接的张量
            has_sudoku = tf.cast(has_sudoku, tf.float32)  # 形状: () → (1,)
            has_sudoku = tf.reshape(has_sudoku, (1,))     # 确保形状为 (1,)
            
            # 处理坐标点：归一化并reshape为 (16, 2)
            points = tf.cast(points, tf.float32)
            points = tf.reshape(points, (16, 2)) / IMG_SIZE  # 归一化到 [0,1]
            
            # 返回多输出格式：图像, (存在标签, 坐标点)
            return img, (has_sudoku, points)
        
        # 训练集
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_has, train_points))
        train_ds = train_ds.shuffle(10000).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        self.train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        # 测试集  
        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_has, test_points))
        test_ds = test_ds.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
        self.test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    def _train_step(self, images, exists_labels, coords_labels):
        """训练步骤"""
        with tf.GradientTape() as tape:
            exists_pred, coords_pred = self.model(images, training=True)
            
            loss = self.loss_fn(
                [exists_labels, coords_labels],
                [exists_pred, coords_pred]
            )
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss
    
    def train(self, epochs):
        """训练循环"""
        for epoch in range(epochs):
            epoch_losses = []
            for batch_data in self.train_ds:
                if len(batch_data) == 3:
                    images, exists_labels, coords_labels = batch_data
                else:
                    images, (exists_labels, coords_labels) = batch_data
                
                loss = self.train_step(images, exists_labels, coords_labels)
                epoch_losses.append(loss)
            
            mean_loss = tf.reduce_mean(epoch_losses)
            print(f"Epoch {epoch}, Loss: {mean_loss:.6f}")

# ==================== 7. 评估指标 ====================
class MultiOutputMetrics:
    """多输出评估指标"""
    def __init__(self):
        self.exist_accuracy = tf.keras.metrics.BinaryAccuracy()
        self.coord_mae = tf.keras.metrics.MeanAbsoluteError()
        self.coord_mse = tf.keras.metrics.MeanSquaredError()
    
    def update_state(self, y_true, y_pred):
        exists_true, coords_true = y_true
        exists_pred, coords_pred = y_pred
        
        # 更新布尔值准确率
        self.exist_accuracy.update_state(exists_true, exists_pred)
        
        # 只对存在的样本计算坐标误差
        mask = tf.cast(exists_true, tf.bool)
        valid_coords_true = tf.boolean_mask(coords_true, mask)
        valid_coords_pred = tf.boolean_mask(coords_pred, mask)
        
        if tf.size(valid_coords_true) > 0:
            self.coord_mae.update_state(valid_coords_true, valid_coords_pred)
            self.coord_mse.update_state(valid_coords_true, valid_coords_pred)
    
    def result(self):
        return {
            'exist_accuracy': self.exist_accuracy.result(),
            'coord_mae': self.coord_mae.result(),
            'coord_mse': self.coord_mse.result()
        }
    
    def reset_states(self):
        self.exist_accuracy.reset_states()
        self.coord_mae.reset_states()
        self.coord_mse.reset_states()

# ==================== 8. 使用示例 ====================

def main():
    # 显存优化
    # 1. 设置GPU内存增长
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     tf.config.experimental.set_memory_growth(gpus[0], True)

    
    """主函数"""
    # 创建模型
    # model = build_texture_enhanced_locator()
    # model = build_light_texture_locator() # 轻量版
    model = build_mini_texture_locator() # 迷你版
    # model = build_light_residual_locator() # 残差版
    # model.summary()
    
    # 创建训练器（可选择使用自适应损失）
    trainer = MultiOutputTrainer(model, use_adaptive_loss=True)
    
    # 训练
    trainer.train(epochs=10)
    
    # 测试预测
    test_image = trainer.test_ds[0][:50][0]
    exists_pred, coords_pred = model.predict(test_image)
    
    print(f"\n测试预测:")
    print(f"存在概率: {exists_pred[0][0]:.3f}")
    print(f"前5个坐标点:")
    for i in range(5):
        print(f"  点{i+1}: x={coords_pred[0][i][0]:.3f}, y={coords_pred[0][i][1]:.3f}")

if __name__ == "__main__":
    main()