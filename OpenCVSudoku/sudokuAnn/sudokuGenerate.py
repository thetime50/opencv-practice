

# 生成器函数 循环从path路径读取图片文件
import os
import random
import cv2
import numpy as np

def image_generator(path = './img/bg', shuffle=True):
    """
    从指定路径循环读取图片文件，返回生成器。
    
    Args:
        path (str): 图片文件夹路径。
        shuffle (bool): 是否打乱图片顺序。
    """
    image_files = [f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if shuffle:
        random.shuffle(image_files)

    # 传入一个图片任意截取
    def random_crop(image, ):
        h, w = image.shape[:2]
        if(random.random() > 0.5):
            dw = w//4
            dh = h//4
            start_point = (random.randint(0, dh), random.randint(0, dw))
            dw2 = dw*3//4
            dh2 = dh*3//4
        else:
            start_point = (0,0)
            dw2 = w//4
            dh2 = h//4
        start_point = np.array(start_point, np.int32)
        points = start_point + np.array([(random.randint(0,dw2), random.randint(0,dh2)),
                                            (random.randint(-dw2,0)+dw2*4, random.randint(0,dh2)),
                                            (random.randint(-dw2,0)+dw2*4, random.randint(-dh2,0)+dh2*4),
                                            (random.randint(0,dw2), random.randint(-dh2,0)+dh2*4)
                                            ], np.int32)
        # 在image上画出points区域并显示
        cv2.polylines(image, [points], isClosed=True, color=(0,255,0), thickness=2)
        # 画start_point
        cv2.circle(image, tuple(start_point), 5, (0, 0, 255), -1)

        resw = w//2
        resh = h//2
        # 将points区域变换到resw*resh的新图片输出
        dst_points = np.array([(0,0), (resw,0), (resw,resh), (0,resh)], np.float32)
        M = cv2.getPerspectiveTransform(points.astype(np.float32), dst_points)
        cropped = cv2.warpPerspective(image, M, (resw, resh))
        return cropped,image
        

    
    while True:
        for image_file in image_files:
            img_path = os.path.join(path, image_file)
            img = cv2.imread(img_path)
            if img is not None:
                yield random_crop(img)
            else:
                print(f"Warning: {img_path} is not a valid image file.")


from .sudokuBg import generate_9x9_coordinates,generate_random_sudoku_background
from dataAugmentation import random_augmentation
from dataAugmentation import random_augmentation

def random_augmentation_seed(img,seed):
    # 锁定随机种子
    random.seed(seed)
    np.random.seed(seed)
    return random_augmentation(img)

'''
1 获取随机背景
    循环获取图片
    随机区域截取变形
2 生成随机数独背景
3 背景添加随机噪声
4 生成随机变换
5 数独背景通过变换添加到背景上
6 生成随机数独数据
    随机数字比例
    生成随机数字 随机颜色
    画出数据
7 随机反色
7 数独按变换添加到背景上
'''
# 60000 10000
# 80000 20000
for i in range(100):
    gen = image_generator()
    bg,src = next(gen)
    cv2.imshow('bg', bg)
    cv2.imshow('src', src)

    sudoku_points = generate_9x9_coordinates()
    # 生成随机数独背景
    sudoku_bg = generate_random_sudoku_background()
    # 背景添加随机噪声
    sudoku_bg = random_augmentation(sudoku_bg)



    seed = random.randint(0, 10000000)

    cv2.waitKey(0)
