# @Time : 2023/2/23 15:43 
# @Author : Li Jiaqi
# @Description :
from albumentations import Compose, OneOf, Normalize, Resize
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, RandomCrop
import platform

LoggingConfig = dict(
    path="log.txt"
)
DataLoaderConfig = dict(
    dataset="E:\\Research\\Datas\\AreialImage\\ArchaeologicalSitesDetection\\georgia_cleaned_all",
    transforms=Compose([
        # RandomCrop(512, 512),
        # OneOf([
        #     HorizontalFlip(True),
        #     VerticalFlip(True),
        #     RandomRotate90(True)
        # ], p=0.75),
        Normalize(mean=(0, 0, 0),
                  std=(255, 255, 255),
                  max_pixel_value=1, always_apply=True),
        Resize(height=420, width=420)
    ]),
    batch_size=1 if platform.system().lower() == 'windows' else 3,
    num_workers=8,
    drop_last=False,  # whether abandon the samples out of batch
    shuffle=True,  # whether to choose the samples in random order
    pin_memory=True  # whether to keep the data in pin memory
)
