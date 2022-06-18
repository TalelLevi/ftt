

DATASET_DIRECTORY = '/home/talel-levi/fft/data/ffcv/imagenette2'

mlops_settings = {
    'ClearML': True,
    'init_settings': {'project_name': 'Fast ViT learning',
                      'task_name': 'NAT with FFCV lr 1e-5 exp 1'}

}

manager_settings = {
    # 'experiment_path': 'temp3',
    # 'active_folds': (i for i in range(1)),
    # 'restore_checkpoints': False,
}

transformations_settings = {
    'imagenet_mean': [0.485, 0.456, 0.406],
    'imagenet_std': [0.229, 0.224, 0.225]
}

loader_settings = {
    'FFCV': True,
    'dataset': {'train': f'{DATASET_DIRECTORY}/train_500_0.50_90.ffcv',
                'val': f'{DATASET_DIRECTORY}/val_500_0.50_90.ffcv',
                'test': f'{DATASET_DIRECTORY}/empty'},
    # 'dataset': {'train': f'/home/talel-levi/fft/data/imagenette2/train',
    #             'val': f'/home/talel-levi/fft/data/imagenette2/val',
    #             'test': f'/home/talel-levi/fft/data/imagenette2/empty'},
    'num_workers': 2,
    'batch_size': 64,
    'distributed': 0,
    'resolution': 224,
    'in_memory': 1  # for ffcv
}

trainer_settings = {
    'num_epochs': 300,
    'checkpoint_path': None,
    'early_stopping': None,
}

model_settings = {

}

optimizer_settings = {
    'learning_rate': 1e-5,
}

scheduler_settings = {

}

loss_functions_settings = {

}

metrics_settings = {

}

preprocess_settings = {

}

postprocess_settings = {

}

## logger handler ##
if mlops_settings['ClearML']:
    from clearml import Task
    task = Task.init(**mlops_settings['init_settings'])
    manager_settings = task.connect(manager_settings, 'manager settings')
    transformations_settings = task.connect(transformations_settings, 'transformations settings')
    loader_settings = task.connect(loader_settings, 'data loader settings')
    trainer_settings = task.connect(trainer_settings, 'trainer settings')
    model_settings = task.connect(model_settings, 'model settings')
    optimizer_settings = task.connect(optimizer_settings, 'optimizer settings')

    scheduler_settings = task.connect(scheduler_settings, 'scheduler settings')
    loss_functions_settings = task.connect(loss_functions_settings, 'loss function settings')
    metrics_settings = task.connect(metrics_settings, 'metrics settings')
    preprocess_settings = task.connect(preprocess_settings, 'preprocessing settings')
    postprocess_settings = task.connect(postprocess_settings, 'postprocessing settings')
    logger = task.get_logger()
