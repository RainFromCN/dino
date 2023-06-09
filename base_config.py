# About loading & saving
BATCH_SIZE = 4
NUM_WORKERS = 3
# TRAIN_SET_DIR = '/root/autodl-tmp/imagenet/train'
TRAIN_SET_DIR = 'D:\datasets\mini\seg_train\seg_train'
OUTPUT_DIR = './output'
STATE_SAVE_FREQ = 1 # 每个epoch保存一个state dict

# About dataset augmentation
NUM_LOCAL_CROPS = 8
LOCAL_CROP_SIZE = 96
GLOBAL_CROP_SIZE = 224
LOCAL_CROP_SCALE = (0.05,0.4)
GLOBAL_CROP_SCALE = (0.4,1)

# About optimization
EPOCHS = 800
USE_AMP = True  # 自动混合精度
OPTIM_METHOD = 'adamw'
CLIP_GRAD_NORM = 3
LEARNING_RATE_BASE = 0.0005
LEARNING_RATE_FINAL = 10 # 学习率warmup的epoch
LEARNING_RATE_WARMUP_EPOCHS = 10
WEIGHT_DECAY_BASE = 0.04
WEIGHT_DECAY_FINAL = 0.4
MOMENTUM_TEACHER_BASE = 0.9
MOMENTUM_TEACHER_FINAL = 1
FREEZE_LAST_LAYER = 1

# About DINO architecture
FEAT_DIM = 65536
DROP_PATH = 0.1
TEMP_STUDENT = 0.1
TEMP_TEACHER_WARMUP = 0.04
TEMP_TEACHER_WARMUP_EPOCHS = 30
TEMP_TEACHER = 0.07

# Misc
SEED = 0
