import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import random
from PIL import ImageFilter, ImageOps
from torch.utils.data import DataLoader
from functools import lru_cache
from torch.utils.data.distributed import DistributedSampler


import base_config as config


class GaussianBlur:
    """
    Apply Gaussian Blur to PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img
        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization:
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.prob = p

    def __call__(self, img):
        if random.random() < self.prob:
            return ImageOps.solarize(img)
        else:
            return img


class DataAugmentation:
    def __init__(self):

        flip_and_color_jitter = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4,0.4,0.2,0.1)], p=0.8),
            T.RandomApply([T.Grayscale(num_output_channels=3)], p=0.2),
        ])

        local_cropper = T.RandomResizedCrop(size=config.LOCAL_CROP_SIZE, 
                                           scale=config.LOCAL_CROP_SCALE)
        global_cropper = T.RandomResizedCrop(size=config.GLOBAL_CROP_SIZE, 
                                             scale=config.GLOBAL_CROP_SCALE)

        normalizer = T.Compose([
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.local_crop = T.Compose([
            local_cropper,
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            normalizer,
        ])

        self.global_crop1 = T.Compose([
            global_cropper,
            flip_and_color_jitter,
            GaussianBlur(1),
            normalizer
        ])

        self.global_crop2 = T.Compose([
            global_cropper,
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            normalizer,
        ]) 

    def __call__(self, image):
        global_crops = [self.global_crop1(image),
                        self.global_crop2(image)]
        local_crops = [self.local_crop(image)
                       for _ in range(config.NUM_LOCAL_CROPS)]
        return global_crops + local_crops


@lru_cache(maxsize=2)
def get_dataset(is_train=True):
    if is_train:
        dataset = ImageFolder(root=config.TRAIN_SET_DIR,
                              transform=DataAugmentation())
    else:
        dataset = ImageFolder(root=config.DEV_SET_DIR,
                              transform=DataAugmentation())
    return dataset


@lru_cache(maxsize=2)
def get_dataloader(dataset, sampler):
    data_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                             num_workers=config.NUM_WORKERS, drop_last=True,
                             pin_memory=True, sampler=sampler)
    return data_loader
