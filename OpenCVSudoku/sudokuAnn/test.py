import numpy as np
import os
SATASET_FILE = os.path.join(os.path.dirname(__file__), 'dataset')
SATASET_FILE_NPY = os.path.join(SATASET_FILE, 'test.npy')

def objectArray(*args):
    res = np.zeros( len(args),object)
    for i,item in enumerate(args):
        res[i] = item
    return res
np.save( # 会覆盖旧文件
    SATASET_FILE_NPY,
    (
        objectArray(['hello'],['good']),
        objectArray(['1'],['2'])
    )
)

(train_set,test_set) = np.load(SATASET_FILE_NPY, allow_pickle=True)
(path_list,_) = train_set
s = path_list[0].numpy().decode('utf-8')
a=1

# ossutil cp -r D:\1024\python\opencv-practice\OpenCVSudoku\sudokuAnn\dataset oss://oss-pai-d8dhll5yzaj0y4jcik-cn-shanghai.oss-cn-shanghai.aliyuncs.com/opencv-practice/OpenCVSudoku/sudokuAnn/dataset/ --region cn-shanghai --include "*.npy"