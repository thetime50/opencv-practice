import cv2
import numpy as np
from scipy.interpolate import interp2d

DEBUG = False

def nonlinear_distortion(image, intensity=0.1,dst = None):
    if DEBUG:
        print(f"nonlinear_distortion intensity={intensity}")
    """
    非线性扭曲效果
    intensity: 扭曲强度 (0.0-1.0)
    """
    h, w = image.shape[:2]
    
    # 创建网格
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 正弦波扭曲
    distortion_x = intensity * 50 * np.sin(2 * np.pi * y / 150)
    distortion_y = intensity * 50 * np.sin(2 * np.pi * x / 150)
    
    # 应用扭曲
    map_x = x + distortion_x
    map_y = y + distortion_y
    
    # 重映射
    distorted = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), 
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,dst=dst)
    return distorted


def nonlinear_distortion_coords(points, image_shape, intensity=0.1):
    """
    给定原图坐标 points，返回它在扭曲后图像中的近似位置
    速度快，但和 nonlinear_distortion(image) 结果可能有轻微偏差
    """
    pts = np.asarray(points, dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]

    dx = intensity * 50 * np.sin(2 * np.pi * ys / 150)
    dy = intensity * 50 * np.sin(2 * np.pi * xs / 150)

    x_new = xs - dx
    y_new = ys - dy

    return np.column_stack((x_new, y_new))

def wave_distortion(image, amplitude=20, frequency=0.05,dst=None):
    if DEBUG:
        print(f"wave_distortion amplitude={amplitude}, frequency={frequency}")
    """波浪扭曲效果"""
    h, w = image.shape[:2]
    
    # 创建网格
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    
    # 波浪效果
    distortion = amplitude * np.sin(2 * np.pi * frequency * x)
    map_x = x
    map_y = y + distortion
    
    distorted = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT, dst=dst)
    return distorted

def wave_distortion_coords(points, image_shape, amplitude=20, frequency=0.05):
    """
    输入原图坐标 points，返回波浪扭曲后的坐标
    与 wave_distortion(image) 结果一致
    """
    pts = np.asarray(points, dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]

    # 在 wave_distortion 中：map_y = y + distortion
    # 输出图 (x', y') 从原图 (x', y' + distortion) 取像素
    # 所以 y' = ys - distortion
    distortion = amplitude * np.sin(2 * np.pi * frequency * xs)
    x_new = xs
    y_new = ys - distortion

    return np.column_stack((x_new, y_new))

def radial_distortion(image, strength=0.0005, center_x=0.5, center_y=0.5, fill_value=0, dst=None):
    if DEBUG:
        print(f"radial_distortion strength={strength}, center=({center_x}, {center_y})")
    """
    径向扭曲效果，保持图片大小不变
    
    Args:
        image: 输入图像
        strength: 扭曲强度
        center_x: 扭曲中心x坐标比例（0-1）
        center_y: 扭曲中心y坐标比例（0-1）
        fill_value: 空白区域填充值（0=黑色，255=白色）
    """
    h, w = image.shape[:2]
    
    # 计算实际中心坐标
    cx = center_x * w
    cy = center_y * h
    
    # 创建归一化网格
    x = (np.arange(w) - cx) / max(cx, w - cx)
    y = (np.arange(h) - cy) / max(cy, h - cy)
    X, Y = np.meshgrid(x, y)
    
    # 计算归一化距离
    R = np.sqrt(X**2 + Y**2)
    
    # 应用径向畸变公式（反向映射）
    # 从目标坐标计算源坐标
    R_source = R * (1 + strength * R**2)
    
    # 计算缩放因子
    scale = R_source / (R + 1e-6)
    scale[R < 1e-6] = 1
    
    # 计算源坐标（从目标坐标反向映射）
    X_source = X * scale
    Y_source = Y * scale
    
    # 转换回像素坐标
    map_x = (X_source * max(cx, w - cx) + cx).astype(np.float32)
    map_y = (Y_source * max(cy, h - cy) + cy).astype(np.float32)
    
    # 使用BORDER_CONSTANT填充空白区域
    distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, 
                         borderMode=cv2.BORDER_TRANSPARENT, dst=dst)
                        #  borderMode=cv2.BORDER_CONSTANT, borderValue=fill_value, dst=dst)
    return distorted

def radial_distortion_coords(x, y, image_shape, strength=0.0005, center_x=0.5, center_y=0.5):
    h, w = image_shape
    cx, cy = center_x * w, center_y * h

    xd = np.asarray(x, np.float32)
    yd = np.asarray(y, np.float32)

    xd_norm = (xd - cx) / max(cx, w - cx)
    yd_norm = (yd - cy) / max(cy, h - cy)
    rd = np.sqrt(xd_norm**2 + yd_norm**2)

    # 反解 r: k*r^3 + r - rd = 0
    r = rd.copy()
    for _ in range(5):  # 牛顿迭代
        r = r - (r * (1 + strength * r**2) - rd) / (1 + 3 * strength * r**2)

    scale = np.where(rd > 1e-6, r / rd, 1.0)

    x_norm = xd_norm * scale
    y_norm = yd_norm * scale

    return x_norm * max(cx, w - cx) + cx, y_norm * max(cy, h - cy) + cy

def batch_distort_coordinates(coords, image_shape, strength=0.0003, center_x=0.5, center_y=0.5):
    """
    批量处理坐标变换
    
    Args:
        coords: N×2的坐标数组 [[x1,y1], [x2,y2], ...]
        image_shape: 图像形状 (h, w)
    
    Returns:
        distorted_coords: 变换后的坐标数组
    """
    x = coords[:, 0]
    y = coords[:, 1]
    distorted_x, distorted_y = radial_distortion_coords(x, y, image_shape, strength,center_x,center_y)
    return np.column_stack([distorted_x, distorted_y])


def random_distortion_field(image, grid_size=20, max_displacement=15, dst=None):
    if DEBUG:
        print(f"random_distortion_field grid_size={grid_size}, max_displacement={max_displacement}")
    """随机扭曲场效果 - 修复版本"""
    h, w = image.shape[:2]
    
    # 创建控制点网格
    grid_x = np.linspace(0, w - 1, grid_size)
    grid_y = np.linspace(0, h - 1, grid_size)
    
    # 生成随机位移
    displacement_x = np.random.uniform(-max_displacement, max_displacement, (grid_size, grid_size))
    displacement_y = np.random.uniform(-max_displacement, max_displacement, (grid_size, grid_size))
    
    # 创建密集网格
    dense_x = np.arange(w)
    dense_y = np.arange(h)
    
    # 分别对x和y方向进行插值
    interp_dx = interp2d(grid_x, grid_y, displacement_x, kind='cubic')
    interp_dy = interp2d(grid_x, grid_y, displacement_y, kind='cubic')
    
    # 生成完整的位移场
    displacement_field_x = interp_dx(dense_x, dense_y).reshape(h, w)
    displacement_field_y = interp_dy(dense_x, dense_y).reshape(h, w)
    
    # 创建映射网格
    map_x, map_y = np.meshgrid(dense_x, dense_y)
    map_x = map_x + displacement_field_x
    map_y = map_y + displacement_field_y
    
    # 应用扭曲
    distorted = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT, dst=dst)
    return distorted,displacement_field_x,displacement_field_y

def random_distortion_coords(points, displacement_field_x, displacement_field_y):
    """
    近似版：一次计算，无迭代
    """
    pts = np.asarray(points, dtype=np.float32)
    xs, ys = pts[:, 0], pts[:, 1]

    h, w = displacement_field_x.shape
    xs_int = np.clip(xs.astype(int), 0, w - 1)
    ys_int = np.clip(ys.astype(int), 0, h - 1)

    dx = displacement_field_x[ys_int, xs_int]
    dy = displacement_field_y[ys_int, xs_int]

    # 这里是正向位移，和 remap 有细微差异
    return np.column_stack((xs - dx, ys - dy))

# def composite_distortion(image, distortions=None):
#     """复合多种扭曲效果"""
#     if distortions is None:
#         distortions = [
#             ('wave', {'amplitude': 15, 'frequency': 0.03}),
#             ('radial', {'strength': 0.0003}),
#             ('nonlinear', {'intensity': 0.05}),
#             ('random', {'grid_size': 20, 'max_displacement': 10})
#         ]
    
#     result = image.copy()
#     for dist_type, params in distortions:
#         if dist_type == 'wave':
#             result = wave_distortion(result, **params)
#         elif dist_type == 'radial':
#             result = radial_distortion(result, **params)
#         elif dist_type == 'nonlinear':
#             result = nonlinear_distortion(result, **params)
#         elif dist_type == 'random':
#             result = random_distortion_field(result, **params)
    
#     return result

def random_perspective_distortion(img,dst):
    src_points = np.int32([[0, 0], [img.shape[1]-1, 0], [img.shape[1]-1, img.shape[0]-1], [0, img.shape[0]-1]])
    h, w = dst.shape[:2]

    rate = np.random.uniform(0.5, 0.99)

    rw = w*rate
    rh = h*rate
    start_point = np.array((np.random.randint(0, w-rw),np.random.randint(0, h-rh)), np.int32)
    dw2 = rw//4
    dh2 = rh//4
    dst_points = start_point + np.array([(np.random.randint(0,dw2), np.random.randint(0,dh2)),
                                            (np.random.randint(-dw2,0)+dw2*4, np.random.randint(0,dh2)),
                                            (np.random.randint(-dw2,0)+dw2*4, np.random.randint(-dh2,0)+dh2*4),
                                            (np.random.randint(0,dw2), np.random.randint(-dh2,0)+dh2*4)
                                            ], np.int32)
    M = cv2.getPerspectiveTransform(src_points.astype(np.float32), dst_points.astype(np.float32))
    warped = cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT, dst=dst)
    return warped, M

def random_perspective_distortion_coords(points, M):
    """
    输入原图坐标 points，返回透视变换后的坐标
    与 random_perspective_distortion(image) 结果一致
    """
    pts = np.asarray(points, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    pts_homogeneous = np.hstack([pts, ones])  # 转为齐次坐标

    transformed_pts = M @ pts_homogeneous.T  # 矩阵乘法
    transformed_pts /= transformed_pts[2, :]  # 齐次坐标归一化

    return transformed_pts[:2, :].T  # 返回二维坐标


def random_distortion_seed(img,dst,points=None,seed=None):
    # 锁定随机种子
    if seed is not None:
        # random.seed(seed)
        np.random.seed(seed)

    if points is None:
        lt = (0,0)
        rb = (img.shape[1]-1, img.shape[0]-1)  # 右下角
        x_step = ( rb[0] - lt[0]) // 3
        y_step = ( rb[1] - lt[1]) // 3
        points = np.array([[lt[0] + i * x_step, lt[1] + j * y_step] for i in range(4) for j in range(4)], dtype=np.float32)

    res_img = img
    res_points = points
    def random_nonlinear():
        nonlocal res_img, res_points
        intensity = np.random.uniform(0.05, 0.15)
        res_img = nonlinear_distortion(img, intensity=intensity)
        res_points = nonlinear_distortion_coords(points, img.shape[:2], intensity=intensity)
    def random_wave():
        nonlocal res_img, res_points
        amplitude = np.random.uniform(10, 30)
        frequency = np.random.uniform(0.0001, 0.004)
        res_img = wave_distortion(img, amplitude=amplitude, frequency=frequency)
        res_points = wave_distortion_coords(points, img.shape[:2], amplitude=amplitude, frequency=frequency)
    def random_radial():
        nonlocal res_img, res_points
        strength = np.random.uniform(0.1, 0.5)
        center_x = np.random.uniform(0.2, 0.8)
        center_y = np.random.uniform(0.2, 0.8)
        res_img = radial_distortion(img, strength=strength, center_x=center_x, center_y=center_y)
        res_points = batch_distort_coordinates(points, img.shape[:2], strength=strength, center_x=center_x, center_y=center_y)
    def random_random():
        nonlocal res_img, res_points
        grid_size = np.random.randint(5, 20)
        max_displacement = np.random.randint(5, 20-grid_size+5)
        res_img,displacement_field_x,displacement_field_y = random_distortion_field(img, grid_size=grid_size, max_displacement=max_displacement)
        res_points = random_distortion_coords(points,displacement_field_x,displacement_field_y)

    
    # 随机选择一种扭曲方式
    distortion_methods = [random_nonlinear, random_wave, random_radial, random_random]
    np.random.choice(distortion_methods)()
    res_img,M = random_perspective_distortion(res_img,dst=dst)
    res_points = random_perspective_distortion_coords(res_points, M)
    return res_img, res_points 

## 批量画点函数
def get_cell_locs(img,points):
    for pt in points:
        cv2.circle(img, tuple(pt.astype(int)), 5, (0, 255, 0), -1)


if __name__ == "__main__":
    # 测试代码
    # img = cv2.imread('../sudoku_puzzle.jpg', cv2.IMREAD_GRAYSCALE)
    # 817*768图片 画一个占比2/3的区域,保存顶点坐标
    img = np.zeros((768, 817,3), dtype=np.uint8)
    lt = (136, 128)  # 左上角
    rb = (680, 640)  # 右下角
    rt = (rb[0], lt[1])  # 右上角
    lb = (lt[0], rb[1])  # 左下角
    cv2.rectangle(img, lt,rb, (255, 255, 255), -1)
    points = np.array([lt, rt, rb, lb], dtype=np.float32)
    # cv2.imshow('Original', img)
    x_step = ( rb[0] - lt[0]) // 3
    y_step = ( rb[1] - lt[1]) // 3
    points = np.array([[lt[0] + i * x_step, lt[1] + j * y_step] for i in range(4) for j in range(4)], dtype=np.float32)
    
     # 测试各个扭曲函数
    
    
    # distorted_img = nonlinear_distortion(img, intensity=0.1)
    # distorted_points = nonlinear_distortion_coords(points, img.shape[:2], intensity=0.1)
    # get_cell_locs(distorted_img, distorted_points)
    # cv2.imshow('Nonlinear Distortion', distorted_img)
    
    # wave_img = wave_distortion(img, amplitude=20, frequency=0.002)
    # wave_points = wave_distortion_coords(points, img.shape[:2], amplitude=20, frequency=0.002)
    # get_cell_locs(wave_img, wave_points)
    # cv2.imshow('Wave Distortion', wave_img)
    
    # radial_img = radial_distortion(img, strength=0.3, center_x=0.3, center_y=0.7)
    # # points = np.array([[0,0], [img.shape[1]-1,0], [img.shape[1]-1,img.shape[0]-1], [0,img.shape[0]-1]], dtype=np.float32)
    # radial_points = batch_distort_coordinates(points, img.shape[:2], strength=0.3, center_x=0.3, center_y=0.7)
    # # cv2.polylines(radial_img, [np.int32(radial_points)], isClosed=True, color=(0, 255, 0), thickness=2)
    # get_cell_locs(radial_img, radial_points)
    # cv2.imshow('Radial Distortion', radial_img)
    
    # random_img,displacement_field_x,displacement_field_y = random_distortion_field(img, grid_size=10, max_displacement=15)
    # random_points = random_distortion_coords(points,displacement_field_x,displacement_field_y)
    # get_cell_locs(random_img, random_points)
    # cv2.imshow('Random Distortion Field', random_img)

    # perspective_img, M = random_perspective_distortion(img, img)
    # perspective_points = random_perspective_distortion_coords(points, M)
    # get_cell_locs(perspective_img, perspective_points)
    # cv2.imshow('Perspective Distortion', perspective_img)
    for _ in range(100):
        img = np.zeros((768, 817,3), dtype=np.uint8)
        cv2.rectangle(img, lt,rb, (255, 255, 255), -1)
        distorted_img, distorted_points = random_distortion_seed(img,img,points=points,seed=None)
        get_cell_locs(distorted_img, distorted_points)
        cv2.imshow('Random Distortion Seed', distorted_img)
        cv2.waitKey(0)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()