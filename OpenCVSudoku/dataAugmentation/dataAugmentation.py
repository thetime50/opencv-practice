import numpy as np
from noise import pnoise2
import numpy as np
from PIL import Image, ImageDraw
import random
import cv2

# 高斯噪声、椒盐噪声
def add_gaussian_noise(img, low = 0, high = 255):
    noise = np.random.randint(low, high, img.shape,np.int16)
    noisy_img = np.clip(img + noise, 0, 255).astype(np.uint8)
    return noisy_img


def perlin_noise_image(img,max = 255, scale=10):
    (w, h) = img.shape
    im = np.zeros((h, w), dtype=np.float32)
    seed = (random.randint(0, 100000),random.randint(0, 100000))
    for y in range(h):
        for x in range(w):
            im[y][x] = pnoise2((x+seed[0])/scale, (y+seed[1])/scale)
    im = ((im - im.min()) / (im.max() - im.min()) * max)
    return np.clip(img + im, 0, 255).astype(img.dtype)

# 随机线条
def draw_random_curves(img, num_curves=2):
    h, w = img.shape[:2]
    for _ in range(num_curves):
        pts = np.array([[random.randint(0, w), random.randint(0, h)] for _ in range(2)], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(img, [pts], False, 255, 
            random.randint(1, 2)) # 随机线宽
    return img


def draw_random_border_lines(img, n, num_lines=2, 
                             sides=('top','bottom','left','right'),
                             thickness_range=(1, 2), rng=None):
    """
    在图像四条边的 n 像素宽度内随机画线。
    - img: np.ndarray，BGR 或灰度
    - n: 边带宽度（像素）
    - num_lines: 画多少条线
    - sides: 要在哪些边画 ('top','bottom','left','right')
    - thickness_range: 线宽随机范围 (min_thick, max_thick)
    - rng: random.Random 实例，便于可复现；None 用全局随机
    """
    if rng is None:
        rng = random

    h, w = img.shape[:2]
    n = int(max(1, min(n, min(h, w))))  # 防越界

    # 生成一个边带内的两端点（保证同一边带）
    def random_line_endpoints(side):
        if side == 'top':
            y1 = y2 = rng.randint(0, n-1)
            x1 = rng.randint(0, w-1)
            x2 = rng.randint(0, w-1)
        elif side == 'bottom':
            y1 = y2 = rng.randint(max(h-n,0), h-1)
            x1 = rng.randint(0, w-1)
            x2 = rng.randint(0, w-1)
        elif side == 'left':
            x1 = x2 = rng.randint(0, n-1)
            y1 = rng.randint(0, h-1)
            y2 = rng.randint(0, h-1)
        elif side == 'right':
            x1 = x2 = rng.randint(max(w-n,0), w-1)
            y1 = rng.randint(0, h-1)
            y2 = rng.randint(0, h-1)
        else:
            raise ValueError('invalid side')
        return (x1, y1), (x2, y2)

    for _ in range(num_lines):
        side = rng.choice(sides)
        p1, p2 = random_line_endpoints(side)
        cv2.line(img, p1, p2, 255, random.randint(*thickness_range), lineType=cv2.LINE_AA)

    return img


def random_augmentation(img):
    """
    随机增强图像，返回增强后的图像。
    - img: np.ndarray，BGR 或灰度
    """

    # 随机选择增强方式
    augmentations = [
        lambda x: add_gaussian_noise(x, 0, 200) if random.random()>0.5 else perlin_noise_image(x,200),
        lambda x: draw_random_curves(x, num_curves=random.randint(1, 3)),
        lambda x: draw_random_border_lines(x, n=3, num_lines=random.randint(1, 3))
    ]

    # 从augmentations中抽卡两个
    for i in random.sample(augmentations, 2):
        if random.random()>0.5:
            img = i(img)
    
    return img

if __name__ == "__main__":
    print(random.randint(1, 2),random.randint(1, 2),random.randint(1, 2),random.randint(1, 2),random.randint(1, 2))
    for _ in range(1):

        im = np.zeros((28,28),np.uint8)
        # cv2.putText(
        #     im,
        #     str(6),
        #     (4,24), #(x,y)
        #     cv2.FONT_ITALIC,
        #     1,255,2,
        #     # bottomLeftOrigin=True
        # )

        # im1 = add_gaussian_noise(im.copy(), 0, 220)
        # im2 = perlin_noise_image(im.copy(),220)
        # im3 = draw_random_curves(im.copy())
        # im4 = draw_random_border_lines(im.copy(),3)

        im1 = random_augmentation(im.copy())
        im2 = random_augmentation(im.copy())
        im3 = random_augmentation(im.copy())
        im4 = random_augmentation(im.copy())


        cv2.namedWindow('title', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('title',1000,250)
        cv2.imshow('title', np.concatenate((im1,im2,im3,im4),axis=1))
        cv2.waitKey(0)