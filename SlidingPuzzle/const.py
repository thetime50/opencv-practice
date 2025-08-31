import os


SATASET_FILE = os.path.join(os.path.dirname(__file__), 'output')
SATASET_FILE_NPY = os.path.join(SATASET_FILE, 'dataset.npy')
MODEL_TEMP_FILE = os.path.join(SATASET_FILE, 'sliding_temp.h5')
# MODEL_TEMP1_FILE = os.path.join(SATASET_FILE, 'sliding_temp_1.h5')
MODEL_FILE = os.path.join(SATASET_FILE, 'sliding.h5')

EPISODES = 300
SIZE_RANGE= (2,10)