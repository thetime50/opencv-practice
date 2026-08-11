import os

IS_DSW = True if os.path.exists('/mnt/workspace') else False

IMG_SIZE = 384  # 缩放到统一大小以保证 batch 内一致
BATCH_SIZE = 64 if IS_DSW else 16