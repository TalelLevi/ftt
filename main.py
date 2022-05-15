import torch
import torchvision
import os
from data_loader.data_loader import Dataset
# from models.vit import ViT
from metrics.accuracy_metric import accuracy_metric
from trainers.train import TorchTrainer as Trainer
from settings import *



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

# train_loader = torch.utils.data.DataLoader(train_dataset,
#                                            batch_size=loader_settings['batch_size'],
#                                            num_workers=loader_settings['num_workers'],
#                                            drop_last=True, shuffle=True, pin_memory=True)
#
# val_loader = torch.utils.data.DataLoader(val_dataset,
#                                          batch_size=loader_settings['batch_size'],
#                                          num_workers=loader_settings['num_workers'],
#                                          drop_last=True, shuffle=False, pin_memory=True)

train_loader = create_train_loader(train_dataset=train_dataset,
                                   num_workers=loader_settings['num_workers'],
                                   batch_size=loader_settings['batch_size'],
                                   in_memory=loader_settings['in_memory'])

val_loader = create_val_loader(val_dataset=val_dataset,
                               num_workers=loader_settings['num_workers'],
                               batch_size=loader_settings['batch_size'],
                               in_memory=loader_settings['in_memory'])

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

metrics = [accuracy_metric()]

optimizer = torch.optim.Adam(model.parameters(),
                             lr=optimizer_settings['learning_rate'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(model,
                  criterion,
                  optimizer,
                  metrics=metrics,
                  device=device,
                  logger=logger)

res = trainer.fit(train_loader,
                  val_loader,
                  num_epochs=trainer_settings['num_epochs'],
                  checkpoint_path=f'model.pth')

