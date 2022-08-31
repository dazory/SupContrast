import math

learning_rate = 0.2
lr_decay_rate = 0.1
eta_min = learning_rate * (lr_decay_rate**3)
warm_epochs = 10
epochs = 500
warmup_to = eta_min + (learning_rate-eta_min) \
            * (1 + math.cos(math.pi * warm_epochs / epochs)) / 2

optimizer = dict(
    type='SGD',
    learning_rate=learning_rate,
    lr_decay_epochs=[350, 400, 450],
    lr_decay_rate=lr_decay_rate,
    weight_decay=0.0001,
    momentum=0.9,
    lr_config=dict(
        cosine=True,
        warm=True,
        warmup_from=0.01,
        warm_epochs=warm_epochs,
        warmup_to=warmup_to
    )
)
runner = dict(
    epochs=epochs,
    trial=0,
)