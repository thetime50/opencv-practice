import numpy as np
import cv2
import random
from typing import List, Tuple

'''
背景生成
'''


# 基础颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
DARK_GRAY = (150, 150, 150)

def generate_9x9_grid_background(border_type: str = "single", bg_type: str = "white") -> np.ndarray:
    """
    生成9×9网格背景，支持内外格线双线效果
    """
    # 创建基础画布
    img = np.ones((504, 504, 3), dtype=np.uint8) * 255
    
    # 设置背景色
    if bg_type == "white":
        pass
    elif bg_type == "3x3_alternate":
        for i in range(3):
            for j in range(3):
                color = (230, 230, 230) if (i + j) % 2 == 0 else (255, 255, 255)
                cv2.rectangle(img, (j*168, i*168), ((j+1)*168, (i+1)*168), color, -1)
    elif bg_type == "row_alternate":
        for i in range(9):
            color = (230, 230, 230) if i % 2 == 0 else (255, 255, 255)
            cv2.rectangle(img, (0, i*56), (504, (i+1)*56), color, -1)
    elif bg_type == "col_alternate":
        for j in range(9):
            color = (230, 230, 230) if j % 2 == 0 else (255, 255, 255)
            cv2.rectangle(img, (j*56, 0), ((j+1)*56, 504), color, -1)
    
    # 绘制网格线
    if border_type != "none":
        if border_type == "double":
            # 绘制9×9细网格双线
            for i in range(1, 9):
                # 垂直线双线
                cv2.line(img, (i*56-1, 0), (i*56-1, 504), (150, 150, 150), 1)  # 左线
                cv2.line(img, (i*56+1, 0), (i*56+1, 504), (150, 150, 150), 1)  # 右线
                # 水平线双线
                cv2.line(img, (0, i*56-1), (504, i*56-1), (150, 150, 150), 1)  # 上线
                cv2.line(img, (0, i*56+1), (504, i*56+1), (150, 150, 150), 1)  # 下线
            
            # 绘制3×3粗网格双线
            for i in range(1, 3):
                # 垂直线双线
                cv2.line(img, (i*168-2, 0), (i*168-2, 504), (0, 0, 0), 2)  # 左粗线
                cv2.line(img, (i*168+2, 0), (i*168+2, 504), (0, 0, 0), 2)  # 右粗线
                # 水平线双线
                cv2.line(img, (0, i*168-2), (504, i*168-2), (0, 0, 0), 2)  # 上粗线
                cv2.line(img, (0, i*168+2), (504, i*168+2), (0, 0, 0), 2)  # 下粗线
            
            # 绘制外边框双线
            # 外框粗线
            cv2.rectangle(img, (1, 1), (502, 502), (0, 0, 0), 3)
            # 内框细线
            cv2.rectangle(img, (5, 5), (498, 498), (0, 0, 0), 1)
            
        else:  # single
            # 单线模式
            for i in range(1, 9):
                cv2.line(img, (i*56, 0), (i*56, 504), (200, 200, 200), 1)
                cv2.line(img, (0, i*56), (504, i*56), (200, 200, 200), 1)
            
            for i in range(1, 3):
                cv2.line(img, (i*168, 0), (i*168, 504), (0, 0, 0), 2)
                cv2.line(img, (0, i*168), (504, i*168), (0, 0, 0), 2)
            
            cv2.rectangle(img, (0, 0), (503, 503), (0, 0, 0), 2)
    
    return img

def generate_3x3_grid_background(border_type: str = "single", bg_type: str = "white") -> np.ndarray:
    """
    生成3×3大格子背景，支持内外格线双线效果
    """
    # 创建基础画布
    img = np.ones((504, 504, 3), dtype=np.uint8) * 255
    
    # 设置背景色
    if bg_type == "white":
        pass
    elif bg_type == "3x3_alternate":
        for i in range(3):
            for j in range(3):
                color = (230, 230, 230) if (i + j) % 2 == 0 else (255, 255, 255)
                cv2.rectangle(img, (j*168, i*168), ((j+1)*168, (i+1)*168), color, -1)
    
    # 绘制网格线
    if border_type != "none":
        if border_type == "double":
            # 绘制3×3粗网格双线
            for i in range(1, 3):
                # 垂直线双线
                cv2.line(img, (i*168-2, 0), (i*168-2, 504), (0, 0, 0), 2)  # 左粗线
                cv2.line(img, (i*168+2, 0), (i*168+2, 504), (0, 0, 0), 2)  # 右粗线
                # 水平线双线
                cv2.line(img, (0, i*168-2), (504, i*168-2), (0, 0, 0), 2)  # 上粗线
                cv2.line(img, (0, i*168+2), (504, i*168+2), (0, 0, 0), 2)  # 下粗线
            
            # 绘制外边框双线
            cv2.rectangle(img, (1, 1), (502, 502), (0, 0, 0), 3)  # 外框粗线
            cv2.rectangle(img, (5, 5), (498, 498), (0, 0, 0), 1)  # 内框细线
            
        else:  # single
            for i in range(1, 3):
                cv2.line(img, (i*168, 0), (i*168, 504), (0, 0, 0), 2)
                cv2.line(img, (0, i*168), (504, i*168), (0, 0, 0), 2)
            
            cv2.rectangle(img, (0, 0), (503, 503), (0, 0, 0), 2)
    
    return img


def generate_background_with_random_highlight(bg_type: str = "white") -> np.ndarray:
    """
    生成随机行列加深的背景
    
    Args:
        bg_type: 基础背景类型
    
    Returns:
        504×504的numpy数组图像
    """
    # 首先生成基础背景
    if bg_type == "3x3":
        img = generate_3x3_grid_background("single", "white")
    else:
        img = generate_9x9_grid_background("single", bg_type)
    
    # 随机选择要加深的行和列
    rows_to_highlight = random.sample(range(9), random.randint(1, 3))
    cols_to_highlight = random.sample(range(9), random.randint(1, 3))
    
    # 加深选中的行
    for row in rows_to_highlight:
        cv2.rectangle(img, 
                     (0, row*56), 
                     (504, (row+1)*56), 
                     DARK_GRAY, -1)
    
    # 加深选中的列
    for col in cols_to_highlight:
        cv2.rectangle(img, 
                     (col*56, 0), 
                     ((col+1)*56, 504), 
                     DARK_GRAY, -1)
    
    return img

def generate_9x9_coordinates() -> List[Tuple[int, int]]:
    """
    生成9×9九宫格的坐标点
    
    Returns:
        81个坐标点的列表，每个点是小格子的中心坐标
    """
    coordinates = []
    for i in range(9):
        for j in range(9):
            # 计算每个小格子的中心坐标
            x = j * 56 + 28  # 56是每个小格子的宽度，28是中心偏移
            y = i * 56 + 28  # 56是每个小格子的高度，28是中心偏移
            coordinates.append((x, y))
    return coordinates


def generate_random_sudoku_background() -> np.ndarray:
    """
    随机生成数独背景图片
    
    Returns:
        504×504的numpy数组图像
    """
    # 随机选择背景类型
    background_types = [
        ("9x9", "white", random.choice(["none", "single", "double"])),
        ("9x9", "3x3_alternate", random.choice(["none", "single", "double"])),
        ("9x9", "row_alternate", random.choice(["none", "single", "double"])),
        ("9x9", "col_alternate", random.choice(["none", "single", "double"])),
        ("3x3", "white", random.choice(["none", "single", "double"])),
        ("3x3", "3x3_alternate", random.choice(["none", "single", "double"])),
        ("random_highlight", "white", "single"),
        ("random_highlight", "3x3", "single")
    ]
    
    # 随机选择一种背景配置
    grid_type, bg_type, border_type = random.choice(background_types)
    
    # 根据选择生成对应的背景
    if grid_type == "9x9":
        return generate_9x9_grid_background(border_type, bg_type)
    elif grid_type == "3x3":
        return generate_3x3_grid_background(border_type, bg_type)
    elif grid_type == "random_highlight":
        return generate_background_with_random_highlight(bg_type)
    
    # 默认返回9x9白色单线边框
    return generate_9x9_grid_background("single", "white")

# 示例使用
if __name__ == "__main__":
    import os
    # 生成9×9坐标点
    coords = generate_9x9_coordinates()
    print(f"生成了 {len(coords)} 个坐标点")
    
    # 生成不同类型的背景
    # backgrounds = [
    #     ("9x9_single_white", generate_9x9_grid_background("single", "white")),
    #     ("9x9_double_3x3_alternate", generate_9x9_grid_background("double", "3x3_alternate")),
    #     ("9x9_none_row_alternate", generate_9x9_grid_background("none", "row_alternate")),
    #     ("9x9_single_col_alternate", generate_9x9_grid_background("single", "col_alternate")),
    #     ("3x3_single_white", generate_3x3_grid_background("single", "white")),
    #     ("3x3_double_3x3_alternate", generate_3x3_grid_background("double", "3x3_alternate")),
    #     ("9x9_single_white", generate_9x9_grid_background("single", "white")),
    #     ("9x9_single_3x3_alternate", generate_9x9_grid_background("single", "3x3_alternate")),
    #     ("9x9_double_row_alternate", generate_9x9_grid_background("double", "row_alternate")),
    #     ("9x9_double_col_alternate", generate_9x9_grid_background("double", "col_alternate")),
    #     ("3x3_double_white", generate_3x3_grid_background("double", "white")),
    #     ("3x3_double_3x3_alternate", generate_3x3_grid_background("double", "3x3_alternate")),
    #     ("random_highlight_white", generate_background_with_random_highlight("white")),
    #     ("random_highlight_3x3_alternate", generate_background_with_random_highlight("3x3_alternate")),
    # ]
    
    # # 保存所有背景图片
    # for name, bg in backgrounds:

        
    #     cv2.imwrite(f"./img/sudoku/bg_{name}.png", bg)
    #     # 打印文件保存绝对路径
    #     print( os.path.abspath(f"./img/sudoku/bg_{name}.png") )

    #     print(f"已保存: {name}.png")=5):
    """测试随机生成多种背景"""
    for i in range(5):
        background = generate_random_sudoku_background()
        cv2.imwrite(f"./img/sudoku/random_background_{i+1}.png", background)
        print(f"已生成随机背景 {i+1}")
    
    print("所有背景图片生成完成!")