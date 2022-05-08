import torch
import torchvision
import os
from data_loader.data_loader import Dataset
# from models.vit import ViT
from metrics.accuracy_metric import accuracy_metric
from trainers.train import TorchTrainer as Trainer
import matplotlib.pyplot as plt
from clearml import Task

task = Task.init(project_name="Fast ViT learning", task_name="testing clearML - acc and loss")
logger = task.get_logger()

# standard imagenet stats
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

from data_loader.data_loader import create_train_loader, create_val_loader

train_dataset = Dataset(os.path.join(os.path.join('data', 'imagenette2'), 'train'), transform,
                        preload_data=False, tqdm_bar=True)
# train_eval_dataset = utils.Dataset(os.path.join(args.data_path, 'train'), TwoCropsTransform(clf_train_transforms),
#                                    preload_data=args.preload_data, tqdm_bar=args.tqdm_bar)
val_dataset = Dataset(os.path.join(os.path.join('data', 'imagenette2'), 'val'), transform,
                      preload_data=False, tqdm_bar=True)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=16,
                                           num_workers=2,
                                           drop_last=True, shuffle=True, pin_memory=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=16,
                                         num_workers=2,
                                         drop_last=True, shuffle=False, pin_memory=True)

# train_loader = create_train_loader(train_dataset=train_dataset)
#
# val_loader = create_val_loader(val_dataset=val_dataset)

# model = ViT(
#     image_size=224,
#     patch_size=32,
#     num_classes=1000,
#     dim=1024,
#     depth=6,
#     heads=16,
#     mlp_dim=2048,
#     dropout=0.1,
#     emb_dropout=0.1
# )

model = torchvision.models.vit_b_16()

criterion = torch.nn.CrossEntropyLoss()
metrics_clf = [accuracy_metric()]
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(model, criterion, optimizer, metrics=metrics_clf, device=device, logger=logger)
res = trainer.fit(train_loader, val_loader, num_epochs=300, checkpoint_path=f'model.pth')
# for y_axis, name in zip(res[1:], ['train_loss', 'train_acc', 'test_loss', 'test_acc']):  # TODO change to plotter
#     plt.plot(y_axis, label=name)
#     plt.savefig(f'plot_{name}.jpg')
#     plt.clf()
