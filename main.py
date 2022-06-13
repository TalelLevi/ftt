import torch
import torchvision
from models.LeViT import LeViT_128S as LeViT
from metrics.accuracy_metric import accuracy_metric
from loss_functions import distilation_loss
from trainers.train import TorchTrainer as Trainer
from settings import *

transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=transformations_settings['imagenet_mean'],
                                     std=transformations_settings['imagenet_std']),
])

if loader_settings['FFCV']:
    from data_loader.data_loader import create_train_loader, create_val_loader
    train_loader = create_train_loader(train_dataset=loader_settings['dataset']['train'],
                                       num_workers=loader_settings['num_workers'],
                                       batch_size=loader_settings['batch_size'],
                                       distributed=loader_settings['distributed'],
                                       in_memory=loader_settings['in_memory'])

    val_loader = create_val_loader(val_dataset=loader_settings['dataset']['val'],
                                   num_workers=loader_settings['num_workers'],
                                   batch_size=loader_settings['batch_size'],
                                   resolution=loader_settings['resolution'],
                                   distributed=loader_settings['distributed'], )

else:
    from data_loader.data_loader import Dataset

    train_dataset = Dataset(loader_settings['dataset']['train'], transform, preload_data=False, tqdm_bar=True)

    val_dataset = Dataset(loader_settings['dataset']['val'], transform, preload_data=False, tqdm_bar=True)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=loader_settings['batch_size'],
                                               num_workers=loader_settings['num_workers'],
                                               drop_last=True, shuffle=True, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=loader_settings['batch_size'],
                                             num_workers=loader_settings['num_workers'],
                                             drop_last=True, shuffle=False, pin_memory=True)

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

# model = torchvision.models.vit_b_16()
model = LeViT(distillation=False, num_classes=10)

criterion = torch.nn.CrossEntropyLoss()
# criterion = distilation_loss()

metrics = [accuracy_metric()]

optimizer = torch.optim.Adam(model.parameters(),
                             lr=optimizer_settings['learning_rate'])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

trainer = Trainer(model=model,
                  loss_fn=criterion,
                  optimizer=optimizer,
                  metrics=metrics,
                  device=device,
                  logger=logger)

res = trainer.fit(train_loader,
                  val_loader,
                  num_epochs=trainer_settings['num_epochs'],
                  checkpoint_path=f'model.pth')
