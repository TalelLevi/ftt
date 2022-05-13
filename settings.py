mlops_settings = {
    'ClearML': True,
    'project_name': 'Fast ViT learning',
    'task_name': 'FA - FFCV Train',

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
    'in_memory': 1
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
