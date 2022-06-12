import torch
import torchvision
import tqdm
import numpy as np

from typing import List
from pathlib import Path

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256


class Dataset(torch.utils.data.Dataset):
    """ a Dataset class for preloading data into memory """
    def __init__(self,
                 path: str,
                 transforms: torchvision.transforms.Compose,
                 preload_data: bool=False,
                 tqdm_bar: bool=False):
        """
        path : str
        """
        super().__init__()
        self.path = path
        self.transforms = transforms
        self.preload_data = preload_data
        self.torchvision_dataset = torchvision.datasets.ImageFolder(path)

        if self.preload_data:
            self.images = []
            self.labels = []

            if tqdm_bar:
                pbar = tqdm(self.torchvision_dataset)
            else:
                pbar = self.torchvision_dataset

            for image, label in pbar:
                self.images.append(image)
                self.labels.append(label)

    def __len__(self):
        return len(self.torchvision_dataset)

    def __getitem__(self, i):
        if self.preload_data:
            img = self.transforms(self.images[i])
            l = self.labels[i]
        else:
            img, l = self.torchvision_dataset.__getitem__(i)
            img = self.transforms(img)
        return img, l

####################################################################


def create_train_loader(train_dataset, num_workers, batch_size, distributed, in_memory):
    this_device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
    train_path = Path(train_dataset)
    assert train_path.is_file()

    res = 224 # self.get_resolution(epoch=0) # TODO find the func
    decoder = RandomResizedCropRGBImageDecoder((res, res))
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device), non_blocking=True)
    ]

    order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
    loader = Loader(train_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=True,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)

    return loader


def create_val_loader(val_dataset, num_workers, batch_size, resolution=224, distributed=0):
    this_device = f'cuda:{0}' if torch.cuda.is_available() else 'cpu'
    print(f'using Cuda ? #### {this_device} ####')
    val_path = Path(val_dataset)
    assert val_path.is_file()
    res_tuple = (resolution, resolution)
    cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device),
        non_blocking=True)
    ]

    loader = Loader(val_dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed)
    return loader