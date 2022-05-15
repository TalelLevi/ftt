
from clearml import Task

mlops_settings = {
    'ClearML': True,
    'init_settings': {'project_name': 'Fast ViT learning',
                      'task_name': 'hyperparam test'}

}

manager_settings = {
    'experiment_path': 'temp3',
    'active_folds': (i for i in range(1)),
    'restore_checkpoints': False,
}

transformations_settings = {

}

loader_settings = {
    'num_workers': 2,
    'batch_size': 16,
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
    'learning_rate': 1e-4,
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
task = Task.init(**mlops_settings['init_settings'])
logger = task.get_logger()
manager_settings = task.connect(manager_settings)
transformations_settings = task.connect(transformations_settings)
loader_settings = task.connect(loader_settings)
trainer_settings = task.connect(trainer_settings)
model_settings = task.connect(model_settings)
optimizer_settings = task.connect(optimizer_settings)

scheduler_settings = task.connect(scheduler_settings)
loss_functions_settings = task.connect(loss_functions_settings)
metrics_settings = task.connect(metrics_settings)
preprocess_settings = task.connect(preprocess_settings)
postprocess_settings = task.connect(postprocess_settings)
