import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import re


SATASET_FILE = os.path.join(os.path.dirname(__file__), 'dataset')
SATASET_FILE_IMG = os.path.join(SATASET_FILE, 'img')
MODEL_FILE = os.path.join(SATASET_FILE, 'sudoku.h5')
SATASET_FILE_NPY = os.path.join(SATASET_FILE, 'sudoku_dataset.npy')

npy_set = None

def masked_keypoints_loss(y_true, y_pred):
    """
    自定义损失函数，当has_sudoku为False时屏蔽keypoints损失
    """
    # y_true格式: [has_sudoku, keypoints]
    has_sudoku = y_true[:, 0]  # 第一列是has_sudoku标签
    keypoints_true = y_true[:, 1:]  # 剩余是keypoints坐标
    keypoints_pred = y_pred
    
    # 计算MSE损失
    mse_loss = tf.keras.losses.mean_squared_error(keypoints_true, keypoints_pred)
    
    # 当has_sudoku为False时，损失为0
    mask = tf.cast(has_sudoku, tf.float32)  # True=1, False=0
    masked_loss = mse_loss * mask
    
    return tf.reduce_mean(masked_loss)
def to_num(s):
    try: return int(s)
    except: return None
class SudokuDetector:
    """数独检测器类"""
    
    def __init__(self, model_path=None):
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path, compile=False)
        else:
            raise Exception('加载模型错误')
    
    def preprocess_image(self, image):
        """预处理图像"""
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"无法读取图像: {image}")
        
        # 转换为灰度
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        
        return image
    
    def detect(self, image):
        """
        检测图像中的数独和关键点
        
        Returns:
            dict: 检测结果
        """
        # 预处理
        original_image = self.preprocess_image(image)
        original_h, original_w = original_image.shape[:2]
        
        # 预处理输入
        # input_image = cv2.resize(original_image, (256, 256))
        input_image =  original_image
        input_image = input_image / 255.0
        input_batch = np.expand_dims(input_image, axis=0)
        
        # 预测
        predict_res = self.model.predict(input_batch, verbose=0)
        has_sudoku_pred = predict_res[0][0]
        keypoints_pred = predict_res[0][1:]
        
        # 处理输出
        has_sudoku_prob = has_sudoku_pred
        has_sudoku = has_sudoku_prob > 0.5
        
        # 转换关键点坐标到原始尺寸
        if has_sudoku:
            keypoints = keypoints_pred.reshape(16, 2)  # 16个点，每个点(x,y)
            keypoints[:, 0] *= original_w  # x坐标
            keypoints[:, 1] *= original_h  # y坐标
            keypoints = keypoints.astype(int)
        else:
            keypoints = None
        
        return {
            'has_sudoku': has_sudoku,
            'confidence': has_sudoku_prob,
            'keypoints': keypoints,
            'keypoints_normalized': keypoints_pred[0].tolist() if has_sudoku else None
        }
    
    def visualize_result(self, image, result, output_path=None):
        """可视化检测结果"""
        img_display = image.copy()
        
        if result['has_sudoku'] and result['keypoints'] is not None:
            # 绘制关键点
            for i, (x, y) in enumerate(result['keypoints']):
                cv2.circle(img_display, (x, y), 2, (0, 255, 0), -1)
                cv2.putText(img_display, str(i+1), (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # 绘制连接线（显示数独网格）
            for i in range(0, 16, 4):
                pts = result['keypoints'][i:i+4].reshape((-1, 1, 2))
                cv2.polylines(img_display, [pts.astype(int)], False, (0, 255, 0), 1)
        
        # 添加文本信息
        status = "Found Sudoku" if result['has_sudoku'] else "No Sudoku"
        color = (0, 255, 0) if result['has_sudoku'] else (0, 0, 255)
        cv2.putText(img_display, f"{status} ({result['confidence']:.3f})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        if output_path:
            cv2.imwrite(output_path, img_display)
        
        return img_display
    
    def detect_test_batch(self,path_list):
        for i,path in enumerate( path_list):
            image = cv2.imread(path)
            result = self.detect(image)
            img_display = self.visualize_result(image,result)
            
            n=to_num( re.split(r'[\\/.]',path)[-2])
            if n is not None:
                global npy_set
                if npy_set is None:
                    npy_set = np.load(SATASET_FILE_NPY, allow_pickle=True)
                train_set, test_set = npy_set
                if n < len(train_set[0]):
                    i=n
                    images_set, has_set, points_set = train_set
                elif n < len(train_set[0]) + len(test_set[0]):
                    i = n-len(train_set[0])
                    images_set, has_set, points_set = test_set
                else:
                    pass
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = detector.visualize_result(image,{
                    'has_sudoku': has_set[i],
                    'confidence': 1 if has_set[i] else 0,
                    'keypoints': points_set[i].astype(np.int16),
                    'keypoints_normalized':None,
                })
                if(has_set[i] and result['has_sudoku']):
                    kps1 = result['keypoints']/image.shape[0]
                    kps2 = points_set[i]/image.shape[0]
                    var_val = np.var(np.concatenate([kps1.ravel(), kps2.ravel()]))
                    print(f"{n} 方差为：{var_val}")
                else:
                    print(f'{n} 没有目标')
                cv2.namedWindow('set_display', cv2.WINDOW_AUTOSIZE)
                cv2.imshow("set_display",img)

            cv2.namedWindow('img_display', cv2.WINDOW_AUTOSIZE)
            cv2.imshow("img_display",img_display)
            cv2.waitKey(0)

path_list = [
    os.path.join(SATASET_FILE_IMG, f"{i//1000:03d}_{i%1000:03d}.png") for i in range(100000-20,100000-10)
] + [
    os.path.join(os.path.dirname(__file__), "../sudoku_puzzle.jpg"),
    os.path.join(os.path.dirname(__file__), "../sudoku_puzzle1.jpg"),
    os.path.join(os.path.dirname(__file__), "../sudoku_puzzle2.jpg"),
    os.path.join(os.path.dirname(__file__), "../sudoku_puzzle3.jpg"),
    os.path.join(os.path.dirname(__file__), "../sudoku_puzzle4.jpg"),
]

if __name__ == '__main__':
    detector = SudokuDetector(MODEL_FILE)
    detector.detect_test_batch(path_list)

    # train_set, test_set = np.load(SATASET_FILE_NPY, allow_pickle=True)
    # train_images, train_has, train_points = train_set
    # for i in range(0,100):
    #     img = cv2.imread(os.path.join(SATASET_FILE_IMG, train_images[i]) )
    #     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     img = detector.visualize_result(img,{
    #         'has_sudoku': train_has[i],
    #         'confidence': 1 if train_has[i] else 0,
    #         'keypoints': train_points[i].astype(np.int16),
    #         'keypoints_normalized':None,
    #     })
    #     cv2.namedWindow('img_display', cv2.WINDOW_AUTOSIZE)
    #     cv2.imshow("img_display",img)
    #     cv2.waitKey(0)
    

