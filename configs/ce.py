_base_ = [
    '/ws/external/configs/_base_/models/resnet50.py',
    '/ws/external/configs/_base_/datasets/cifar10.py',
    '/ws/external/configs/_base_/schedules/schedule_1x.py',
    '/ws/external/configs/_base_/default_runtime.py'
]

data = dict(
    batch_size=64,
    num_workers=0,
)
optimizer = dict(
    learning_rate=0.8,
)
lr_config = dict(
    cosine=True,
    warm=True,
    warmup_from=0.01,
    warm_epochs=10
)
runner = dict(epochs=8)

log_config = dict(
    loggers=[
        dict(type='wandb_logger',
             init_kwargs=dict(project="supcontrast", entity="hong-dasol",
                              config=dict()),
             interval=500,
             )
    ]
)
