from data import get_dataset, get_dataloader
from dino import DINO
from utils import cancel_gradients_last_layer
import config
import torch
import logging, os
import time
import argparse


parser = argparse.ArgumentParser(prog="DINO")
parser.add_argument("--batch_size", type=int)
parser.add_argument("--epoch", type=int)
args = parser.parse_args()
if args.batch_size: config.BATCH_SIZE = args.batch_size
if args.epoch: config.EPOCHS = args.epoch

# 创建输出目录
os.mkdir(config.OUTPUT_DIR, exist_ok=True)
os.mkdir(config.LOGGING_DIR, exist_ok=True)
os.mkdir(config.CHECKPOINT_DIR, exist_ok=True)
# 设置Logging
logging.basicConfig(filename=os.path.join(config.LOGGING_DIR, 'train.log'), 
                    level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')

# 设置模型, dataloader以及优化器
data_loader = get_dataloader(get_dataset(is_train=True))
model = DINO(feat_dim=config.FEAT_DIM, 
                drop_path=config.DROP_PATH)
optimizer = torch.optim.AdamW(model.param_groups())
scaler = torch.cuda.amp.GradScaler()
if "cuda" in config.DEVICE and config.DEVICE_COUNT > 1:
    # 使用Data Parallel
    model = torch.nn.DataParallel(model)
model = model.to(config.DEVICE)

iter = 0
loss_record = [] # 记录loss

for epoch in range(config.EPOCHS):
    temp_std = config.TEMP_STUDENT
    temp_tea = config.TEMP_TEACHER_SCHEDULE[epoch]

    for images, _ in data_loader:
        start = time.time()
        # 将图片迁移到设备中
        images = [img.to(config.DEVICE) for img in images]

        # 查找本次迭代的参数
        lr = config.LR_SCHEDULE[iter]
        wd = config.WD_SCHEDULE[iter]

        # 设置本轮的learning rate和weight decay
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
            if i == 0: param_group['weight_decay'] = wd

        # 学生网络使用优化器进行更新
        optimizer.zero_grad()
        loss = model(images, temp_std, temp_tea)
        if config.USE_AMP and "cuda" in config.DEVICE:
            # 使用自动混合精度
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.student.parameters(),
                                            max_norm=config.CLIP_GRAD_NORM)
            if epoch < config.FREEZE_LAST_LAYER:
                cancel_gradients_last_layer(model.dino_loss)
            scaler.step(optimizer)
            scaler.update()
        else:
            # 不使用自动混合精度
            loss.backward()
            torch.nn.utils.clip_grad.clip_grad_norm_(
                model.student.parameters(),
                max_norm=config.CLIP_GRAD_NORM)
            if epoch < config.FREEZE_LAST_LAYER:
                cancel_gradients_last_layer(model.dino_loss)
            optimizer.step()

        # 教师网络使用EMA进行更新
        with torch.no_grad():
            momentum_teacher = config.MOMENTUM_TEACHER_SCHEDULE[iter]
            for param_student, param_teacher in zip(model.student.parameters(), 
                                                    model.teacher.parameters()):
                param_teacher.data.mul_(momentum_teacher)
                param_teacher.data.add_(param_student.data * (1 - momentum_teacher))

        iter += 1
        print(f"Iter-{iter}: \t {config.BATCH_SIZE / (time.time() - start):.5f} images/s")
        loss_record.append(loss.item())
    
    sum_loss_last_epoch = sum(loss_record[-config.NITER_PER_EP])
    mean_loss_last_epoch = (sum_loss_last_epoch / config.NITER_PER_EP)
    logging.info(f"Epoch[{epoch}/{config.EPOCHS}]:\t" 
                    f"loss={mean_loss_last_epoch}")
    
    if epoch % config.STATE_SAVE_FREQ == 0:
        torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f'{epoch:04}.pth'))
