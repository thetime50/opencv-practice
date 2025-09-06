

# 生成器函数 循环从path路径读取图片文件
import os
import random
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import cv2
import numpy as np
import sys
from PIL import Image
from const import SATASET_FILE,\
    SATASET_FILE_IMG,\
    SATASET_FILE_NPY,\
    MODEL_TEMP_FILE,\
    MODEL_TEMP1_FILE,\
    MODEL_FILE

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

print("启动")


def image_generator(path = './img/bg',width=384,height=384, shuffle=True):
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
        cw = width
        ch = height
        gep_w = cw//4
        gep_h = ch//4
        start_point = np.array((random.randint(0, w-2*gep_w-cw),random.randint(0, h-2*gep_h-ch)), np.int32)
        temp = min(w-start_point[0]-2*gep_w-cw,h-start_point[1]-2*gep_h-ch)
        end_range = np.array((2*gep_w+cw,2*gep_h+ch), np.int32) +random.randint(0,temp)
        points = start_point + np.array([(random.randint(0,gep_w), random.randint(0,gep_h)),
                                            (random.randint(-gep_w,0)+end_range[0], random.randint(0,gep_h)),
                                            (random.randint(-gep_w,0)+end_range[0], random.randint(-gep_h,0)+end_range[1]),
                                            (random.randint(0,gep_w), random.randint(-gep_h,0)+end_range[1])
                                            ], np.int32)
        # # 在image上画出points区域并显示
        # cv2.polylines(image, [points], isClosed=True, color=(0,255,0), thickness=2)
        # # # 画start_point
        # cv2.circle(image, tuple(start_point), 5, (0, 0, 255), -1)

        resw = width
        resh = height
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
                # print(f"Warning: {img_path} is not a valid image file.")
                try:
                    pil_img = Image.open(img_path)
                    img = np.array(pil_img)
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    yield random_crop(img)
                except Exception as e:
                    print(f"PIL读取也失败: {image_file} {e}")



def add_border(image, lrtb, color=(255, 255, 255,255)):
    """
    给图片添加边框 并转为rgba
    
    Args:
        image: 输入图像
        lrtb: 边框宽度 [左, 右, 上, 下] 或 统一宽度
        color: 边框颜色 (B, G, R)
    
    Returns:
        带边框的图像
    """
    # 处理lrtb参数
    if isinstance(lrtb, int):
        left = right = top = bottom = lrtb
    elif len(lrtb) == 4:
        left, right, top, bottom = lrtb
    else:
        raise ValueError("lrtb should be int or list of 4 integers")
    
    # 获取图像尺寸
    h, w = image.shape[:2]
    
    # 计算新图像尺寸
    new_h = h + top + bottom
    new_w = w + left + right
    
    # # 创建新图像并填充边框颜色
    # if len(image.shape) == 3:  # 彩色图像
    #     bordered = np.full((new_h, new_w, 3), color, dtype=image.dtype)
    # else:  # 灰度图像
    #     bordered = np.full((new_h, new_w), color[0], dtype=image.dtype)
    bordered = np.full((new_h, new_w, 4), color, dtype=image.dtype)
    
    # 将原图像放入中心
    bordered[top:top+h, left:left+w,:3] = image
    # bordered[top:top+h, left:left+w,3] = 255
    
    return bordered

def graduatedBorder(image,border=10,range_=[0,255]):
    j, k = range_  # 起始和结束值
    # 2px透明避免透明度通道混合产生边界线
    image[:, 0, 3] = range_[0]
    image[:, -1, 3] = range_[0]
    image[0, :, 3] = range_[0]
    image[-1, :, 3] = range_[0]
    for i in range(1,border):
        val = j + (k - j) * i / (border - 1)
        image[i, i:-i,3] = val       # 上边
        image[-i-1, i:-i,3] = val    # 下边
        image[i:-i, i,3] = val       # 左边
        image[i:-i, -i-1,3] = val    # 右边
    return image

def mean_bg(bg,temp_bg,points):
    '''设置背景色到平均值'''
    xmin = int(min(points[:,0]))
    xmax = int(max(points[:,0]))
    ymin = int(min(points[:,1]))
    ymax = int(max(points[:,1]))
    mean_rgb = np.mean(bg[xmin:xmax, ymin:ymax])/255 #, axis=(0, 1))
    if(mean_rgb>0.5):
        temp_bg[:,:,:3]=temp_bg[:,:,:3]*mean_rgb
    else:
        temp_bg[:,:,:3]=255 -temp_bg[:,:,:3]*(1-mean_rgb)
    return temp_bg

def overlay_rgba_on_rgb(rgb_bg, rgba_img):
    """
    将RGBA图片叠加到RGB背景上
    
    Args:
        rgba_img: 带透明度的前景图像 (H, W, 4)
        rgb_bg: RGB背景图像 (H, W, 3)
    
    Returns:
        叠加后的RGB图像
    """
    # 确保背景图像是RGB
    if len(rgb_bg.shape) == 2:
        rgb_bg = cv2.cvtColor(rgb_bg, cv2.COLOR_GRAY2BGR)
    
    # 分离RGBA通道
    foreground = rgba_img[:, :, :3]  # RGB通道
    alpha = rgba_img[:, :, 3] / 255.0  # Alpha通道 (0-1)
    
    # 将alpha通道扩展为3通道
    alpha = np.stack([alpha, alpha, alpha], axis=-1)
    
    # 计算叠加结果
    result = foreground * alpha + rgb_bg * (1 - alpha)
    
    return result.astype(np.uint8)


'''
填充数字数据
'''

from tensorflow.keras.datasets import mnist

# 加载MNIST数据
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

def draw_mnist_on_image(img, points, fill_rate=0.3):
    """
    根据填充率将MNIST数字随机绘制到图像上
    
    Args:
        img: 背景图像
        points: 起始坐标列表 [(x1,y1), (x2,y2), ...]
        fill_rate: 数据填充率 (0.2-0.5)
    """
    h, w = img.shape[:2]
    
    # 计算需要绘制的数字数量
    num_points = len(points)
    num_to_draw = int(num_points * fill_rate)
    
    # 随机选择要绘制的数字和位置
    indices = np.random.choice(len(trainData), num_to_draw, replace=False)
    point_indices = np.random.choice(num_points, num_to_draw, replace=False)
    
    for i, point_idx in enumerate(point_indices):
        # 获取MNIST数字和对应的坐标
        digit = trainData[indices[i]]
        # digit随机放大1-2倍
        scale = random.uniform(1.0, 2.0)
        offset = int((28*2 - 28*scale)//2)
        digit = 255 - digit
        digit = cv2.resize(digit, (int(28*scale), int(28*scale)))
        # 变成rgba
        digit = np.stack([digit]*3 + [digit], axis=-1)
        digit[:,:,3] = 255-digit[:,:,3]
        x, y = points[point_idx]
        x+= offset
        y+= offset
        # 确保坐标在图像范围内
        if x + digit.shape[1] <= w and y + digit.shape[0] <= h:
            # 将数字绘制到图像上（黑色数字）
            # img[y:y+digit.shape[0], x:x+digit.shape[1]] = np.minimum(img[y:y+digit.shape[0], x:x+digit.shape[1]], digit)
            
            alpha = digit[..., 3] / 255.0  # 获取 alpha 通道并归一化
            img[y:y+digit.shape[0], x:x+digit.shape[1], :3] = (
                alpha[..., np.newaxis] * digit[..., :3] +
                (1 - alpha[..., np.newaxis]) * img[y:y+digit.shape[0], x:x+digit.shape[1], :3]
            )
    
    return img



datainfo=[
    {'text':'FONT_HERSHEY_SIMPLEX',         'font':cv2.FONT_HERSHEY_SIMPLEX,'count':10000},
    # {'text':'FONT_HERSHEY_PLAIN',           'font':cv2.FONT_HERSHEY_PLAIN ,'count':10000},
    {'text':'FONT_HERSHEY_DUPLEX',          'font':cv2.FONT_HERSHEY_DUPLEX ,'count':10000},
    {'text':'FONT_HERSHEY_COMPLEX',         'font':cv2.FONT_HERSHEY_COMPLEX ,'count':10000},
    {'text':'FONT_HERSHEY_TRIPLEX',         'font':cv2.FONT_HERSHEY_TRIPLEX ,'count':10000},
    # {'text':'FONT_HERSHEY_COMPLEX_SMALL',   'font':cv2.FONT_HERSHEY_COMPLEX_SMALL ,'count':10000},
    {'text':'FONT_HERSHEY_SCRIPT_SIMPLEX',  'font':cv2.FONT_HERSHEY_SCRIPT_SIMPLEX ,'count':10000},
    {'text':'FONT_HERSHEY_SCRIPT_COMPLEX',  'font':cv2.FONT_HERSHEY_SCRIPT_COMPLEX ,'count':10000},
    {'text':'FONT_ITALIC',                  'font':cv2.FONT_ITALIC ,'count':10000},
]

thickness=[2,4,6,8]
srcshap = [28*2,28*2]
digitSrc = []
for index,item in enumerate(datainfo):
    row = []
    print(index,item)
    for num in range(10):
        thi = []
        for t in thickness:
            im = np.zeros(srcshap,np.uint8)
            cv2.putText(
                im,
                str(num),
                (8,48), #(x,y)
                item['font'],
                2,255,t,
                # bottomLeftOrigin=True
            )
            thi.append(im)
        row.append(thi)
    digitSrc.append(row)

digitSrc = np.asarray(digitSrc,dtype = 'uint8')


def draw_digits_from_src(img, points, fill_rate=0.3):
    """
    从digitSrc数组中随机选取数字绘制到图像上
    
    Args:
        img: 背景图像
        points: 起始坐标列表 [(x1,y1), (x2,y2), ...]
        digitSrc: 数字源数组 [fontFace, digit, thickness]
        fill_rate: 数据填充率 (0.2-0.5)
    """
    global digitSrc
    h, w = img.shape[:2]
    
    # 计算需要绘制的数字数量
    num_points = len(points)
    num_to_draw = int(num_points * fill_rate)
    
    # 随机选择要绘制的数字和位置
    digitSrcShape = digitSrc.shape
    indices = [ (np.random.randint(0, digitSrcShape[0]),
                 np.random.randint(0, digitSrcShape[1]),
                    np.random.randint(0, digitSrcShape[2])) for _ in range(num_to_draw)]
    point_indices = np.random.choice(num_points, num_to_draw, replace=False)
    
    for i, point_idx in enumerate(point_indices):
        # 获取数字数据和对应的坐标
        digit = digitSrc[indices[i][0], indices[i][1], indices[i][2]]

        # digit随机放大0.5-1倍
        scale = random.uniform(0.5,1)
        offset = int(28*2 * (1 - scale)//2)
        digit = 255 - digit
        digit = cv2.resize(digit, (int(28*2*scale), int(28*2*scale)))
        # 变成rgba
        digit = np.stack([digit]*3 + [digit], axis=-1)
        digit[:,:,3] = 255-digit[:,:,3]
        x, y = points[point_idx]
        x+= offset
        y+= offset

        
        if x + digit.shape[1] <= w and y + digit.shape[0] <= h:
            alpha = digit[..., 3] / 255.0  # 获取 alpha 通道并归一化
            img[y:y+digit.shape[0], x:x+digit.shape[1], :3] = (
                alpha[..., np.newaxis] * digit[..., :3] +
                (1 - alpha[..., np.newaxis]) * img[y:y+digit.shape[0], x:x+digit.shape[1], :3]
            )
        
def random_draw_digits(img, points, fill_rate=0.3):
    if random.random() > 0.5:
        draw_mnist_on_image(img, points, fill_rate)
    else:
        draw_digits_from_src(img, points, fill_rate)

    return img


from sudokuBg import generate_9x9_coordinates,generate_3x3_coordinates,generate_random_sudoku_background
from dataAugmentation import random_augmentation2,random_distortion_seed,get_cell_locs
from tqdm import tqdm

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
8 数独按变换添加到背景上

1 图片尺寸调整
    960*540 -> 384*384
    28*2*9=504 28*9=252
2 噪声调整
'''
NO_SUDOKU_RATE = 0.12

# 60000 10000
# 80000 20000
trainDataCount = 80000
testDataCount = 20000
# trainDataCount = 8000
# testDataCount = 2000
res_trainData = []
res_trainHas = []
res_trainPoints = []
res_testData = []
res_testHas = []
res_testPoints = []
temp_bg = None
for i in tqdm(range(trainDataCount + testDataCount), desc="处理进度"):
    gen = image_generator()
    bg,src = next(gen)
    # cv2.imshow('bg', bg)
    # cv2.imshow('src', src)
    # cv2.waitKey(0)
    # continue

    has_sudoku = False

    key_points = np.array( [[0,0]]*16,np.int16)

    if random.random()>NO_SUDOKU_RATE:
        has_sudoku = True
        sudoku_points = generate_9x9_coordinates() # 画数字用
        key_points = generate_3x3_coordinates() # 关键坐标点
        # 生成随机数独背景
        sudoku_bg = generate_random_sudoku_background()
        lrtb = [random.randint(5,80) for _ in range(4)]
        sudoku_points += np.array([lrtb[0], lrtb[2]])
        key_points += np.array([lrtb[0], lrtb[2]])
        sudoku_bg = add_border(sudoku_bg, lrtb)
        # sudoku_bg四条边设为透明
        graduated = random.randint(190, 255)
        sudoku_bg[:,:, 3] = graduated
        sudoku_bg = graduatedBorder(sudoku_bg,15,[0,graduated])
        # 背景添加随机噪声
        # noic_img = 255 - random_augmentation2(np.zeros(shape= sudoku_bg.shape[:2], dtype=np.uint8))
        # noic_img = np.stack([noic_img]*3, axis=-1)
        # sudoku_bg = np.minimum(sudoku_bg,noic_img)
        sudoku_bg = random_augmentation2(sudoku_bg)
        random_draw_digits(sudoku_bg, sudoku_points, fill_rate=random.uniform(0.2, 0.5))
        
        # 扭曲变换图片
        # 尺寸为bg的rgba图片填充(0,0,0,0)
        temp_bg = np.zeros((bg.shape[0], bg.shape[1], 4), dtype=np.uint8)
        seed = random.randint(0, 10000000)
        _,key_points = random_distortion_seed(sudoku_bg,temp_bg,key_points, seed=seed)

        # 设置temp_bg背景色到平均值
        temp_bg = mean_bg(bg,temp_bg,key_points)

        bg = overlay_rgba_on_rgb(bg,temp_bg)

    # bg 保存图片,文件名为000_000.png
    fileName = f"{i//1000:03d}_{i%1000:03d}.png"
    cv2.imwrite(os.path.join(SATASET_FILE_IMG, fileName), bg)
    key_points = key_points.astype(np.float32)

    if(i < trainDataCount):
        res_trainData.append(fileName)
        res_trainHas.append(has_sudoku)
        res_trainPoints.append(key_points)
    else:
        res_testData.append(fileName)
        res_testHas.append(has_sudoku)
        res_testPoints.append(key_points)

    # # get_cell_locs(bg,key_points)
    # cv2.imshow('src', src)
    # cv2.imshow('bg', bg)
    # if temp_bg is not None: cv2.imshow('temp_bg', temp_bg)
    # # cv2.imshow('temp_bg1', temp_bg[:, :, 3])
    # cv2.waitKey(0)

def objectArray(*args):
    res = np.zeros( len(args),object)
    for i,item in enumerate(args):
        res[i] = item
    return res

print(f'res_trainData len:{len(res_trainData)}, res_testData len:{len(res_testData)}')
print(f'保存数据文件')
np.save( # 会覆盖旧文件
    SATASET_FILE_NPY,
    (
        objectArray(res_trainData, res_trainHas,res_trainPoints),
        objectArray(res_testData, res_testHas,res_testPoints)
    )
)
print(f'保存结束')
# 处理进度: 100%|██████████████████████| 100000/100000 [4:02:15<00:00,  6.88it/s]
# 处理进度: 100%|██████████████████████| 100000/100000 [3:27:02<00:00,  8.05it/s]