_base_ = [
    '/ws/external/configs/_base_/models/resnet50_supcon.py',
    '/ws/external/configs/_base_/datasets/cifar10.py',
    '/ws/external/configs/_base_/schedules/schedule_1x.py',
    '/ws/external/configs/_base_/default_runtime.py'
]

data = dict(
    batch_size=64,
    num_workers=0# 16,
)
model=dict(method='SupCon', temp=0.1)
optimizer = dict(
    learning_rate=0.5,
)
lr_config = dict(
    cosine=True,
    warm=False,
    warmup_from=0.01,
    warm_epochs=10
)
runner = dict(epochs=8)

