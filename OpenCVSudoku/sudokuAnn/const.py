import os

SATASET_FILE = os.path.join(os.path.dirname(__file__), 'dataset')
SATASET_FILE_IMG = os.path.join(SATASET_FILE, 'img')
SATASET_FILE_NPY = os.path.join(SATASET_FILE, 'sudoku_dataset.npy')
MODEL_TEMP_FILE = os.path.join(SATASET_FILE, 'sudoku_temp.h5')
MODEL_TEMP1_FILE = os.path.join(SATASET_FILE, 'sudoku_temp_1.h5')
MODEL_FILE = os.path.join(SATASET_FILE, 'sudoku.h5')


D_MODEL_TEMP_FILE = os.path.join(SATASET_FILE, 'd_sudoku_temp.h5')
D_MODEL_TEMP1_FILE = os.path.join(SATASET_FILE, 'd_sudoku_temp_1.h5')
D_MODEL_FILE = os.path.join(SATASET_FILE, 'd_sudoku.h5')

MODEL_TEMP_FILE_ATT = os.path.join(SATASET_FILE, 'd_sudoku_temp.h5')
MODEL_TEMP1_FILE_ATT = os.path.join(SATASET_FILE, 'd_sudoku_temp_1.h5')
MODEL_FILE_ATT = os.path.join(SATASET_FILE, 'd_sudoku.h5')