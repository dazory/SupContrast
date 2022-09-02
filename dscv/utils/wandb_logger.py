class WandbLogger():
    def __init__(self,
                 init_kwargs=None,
                 interval=10,):
        super(WandbLogger, self).__init__()
        self.wandb = None
        self.init_kwargs = init_kwargs
        self.interval = interval

        self.import_wandb()

    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError('Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    # run
    def before_run(self):
        if self.wandb is None:
            self.import_wandb()
        if self.init_kwargs:
            self.wandb.init(**self.init_kwargs)
        else:
            self.wandb.init()

    def after_run(self):
        self.wandb.finish()

    # epoch
    def before_train_epoch(self):
        pass

    def after_train_epoch(self):
        pass

    def before_val_epoch(self):
        pass

    def after_val_epoch(self):
        pass

    # iter
    def before_train_iter(self):
        pass

    def after_train_iter(self):
        pass

    def before_val_iter(self):
        pass

    def after_val_iter(self):
        pass