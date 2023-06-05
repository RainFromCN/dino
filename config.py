from base_config import *
from utils import cosine_scheduler
import numpy as np
import torch
from data import get_dataset, get_dataloader
import os


CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoint')
LOGGING_DIR = os.path.join(OUTPUT_DIR, 'logging')

torch.random.manual_seed(SEED)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_COUNT = torch.cuda.device_count()
NITER_PER_EP = len(get_dataloader(get_dataset(), None))
USE_AMP = USE_AMP and torch.cuda.is_available()

LR_SCHEDULE = cosine_scheduler(
    base_value=LEARNING_RATE_BASE, final_value=LEARNING_RATE_FINAL, 
    epochs=EPOCHS, niter_per_ep=NITER_PER_EP, 
    warmup_epochs=LEARNING_RATE_WARMUP_EPOCHS, start_warmup_value=0)

WD_SCHEDULE = cosine_scheduler(
    base_value=WEIGHT_DECAY_BASE, final_value=WEIGHT_DECAY_FINAL,
    epochs=EPOCHS, niter_per_ep=NITER_PER_EP)

MOMENTUM_TEACHER_SCHEDULE = cosine_scheduler(
    base_value=MOMENTUM_TEACHER_BASE, final_value=MOMENTUM_TEACHER_FINAL,
    epochs=EPOCHS, niter_per_ep=NITER_PER_EP)

TEMP_TEACHER_SCHEDULE = np.concatenate((
    np.linspace(TEMP_TEACHER_WARMUP, TEMP_TEACHER, TEMP_TEACHER_WARMUP_EPOCHS),
    np.ones(EPOCHS - TEMP_TEACHER_WARMUP_EPOCHS) * TEMP_TEACHER))
