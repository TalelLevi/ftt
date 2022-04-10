import torch
import torchvision
import tqdm

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

