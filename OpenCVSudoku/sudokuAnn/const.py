import os
from build_model import IS_DSW

SATASET_FILE = os.path.join(os.path.dirname(__file__), 'dataset')

SATASET_FILE = os.path.join(os.path.dirname(__file__), 'dataset')
DATASET_FILE = os.path.join('/mnt/oss/oss-pai',*SATASET_FILE.split(os.sep)[3:]) if IS_DSW else SATASET_FILE

SATASET_FILE_IMG = os.path.join(DATASET_FILE, 'img')
SATASET_FILE_NPY = os.path.join(DATASET_FILE, 'sudoku_dataset.npy')
MODEL_TEMP_FILE = os.path.join(SATASET_FILE, 'sudoku_temp.h5') # 训练中 中断的模型 
MODEL_TEMP1_FILE = os.path.join(SATASET_FILE, 'sudoku_temp_1.h5') # 训练后 备份中断模型
MODEL_FILE = os.path.join(SATASET_FILE, 'sudoku.h5') # 训练模型


D_MODEL_TEMP_FILE = os.path.join(SATASET_FILE, 'd_sudoku_temp.h5')
D_MODEL_TEMP1_FILE = os.path.join(SATASET_FILE, 'd_sudoku_temp_1.h5')
D_MODEL_FILE = os.path.join(SATASET_FILE, 'd_sudoku.h5')

MODEL_TEMP_FILE_ATT = os.path.join(SATASET_FILE, 'd_sudoku_temp.h5')
MODEL_TEMP1_FILE_ATT = os.path.join(SATASET_FILE, 'd_sudoku_temp_1.h5')
MODEL_FILE_ATT = os.path.join(SATASET_FILE, 'd_sudoku.h5')

HEAT_MODEL_TEMP_FILE = os.path.join(SATASET_FILE, 'sudoku_heat_temp.h5')
HEAT_MODEL_TEMP1_FILE = os.path.join(SATASET_FILE, 'sudoku_heat_temp_1.h5')
HEAT_MODEL_FILE = os.path.join(SATASET_FILE, 'sudoku_heat.h5')
