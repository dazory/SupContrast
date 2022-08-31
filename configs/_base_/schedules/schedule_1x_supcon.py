learning_rate = 0.05
optimizer = dict(
    type='SGD',
    learning_rate=learning_rate,
    lr_decay_epochs=[700, 800, 900],
    lr_decay_rate=0.1,
    weight_decay=0.0001,
    momentum=0.9,
    lr_config=dict(
        cosine=False,
        warm=True,
        warmup_from=0.01,
        warm_epochs=10,
        warmup_to=learning_rate
    )
)
runner = dict(
    epochs=500,
    trial=0,
)