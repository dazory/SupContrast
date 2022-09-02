import tensorboard_logger as tb_logger
from dscv.utils.wandb_logger import WandbLogger

class Logger():
    def __init__(self, cfg):
        super(Logger, self).__init__()
        self.print_freq = cfg.print_freq
        self.save_freq = cfg.save_freq

        self.tb_logger = None
        self.wandb_logger = None
        self.data = dict()

        self._epoch = 0
        self._iter = 0

        for logger in cfg.loggers:
            if logger.type == 'tb_logger':
                logger = tb_logger.Logger(logdir=logger.tb_folder, flush_secs=2)
                self.tb_logger = logger
            elif logger.type == 'wandb_logger':
                logger = WandbLogger(init_kwargs=logger.init_kwargs,
                                     interval=logger.interval)
                self.wandb_logger = logger

    def update_data(self, data_dict):
        self.data.update(data_dict)

    # run
    def before_run(self):
        if self.wandb_logger:
            self.wandb_logger.before_run()

    def after_run(self):
        if self.wandb_logger:
            self.wandb_logger.after_run()

    # epoch
    def before_train_epoch(self):
        if self.wandb_logger:
            self.wandb_logger.before_train_epoch()
        if self.tb_logger:
            for key, value in self.data.items():
                self.tb_logger.log_value(key, value, self._epoch)

    def after_train_epoch(self):
        if self.wandb_logger:
            self.wandb_logger.after_train_epoch()

    def before_val_epoch(self):
        if self.wandb_logger:
            self.wandb_logger.before_val_epoch()

    def after_val_epoch(self):
        if self.wandb_logger:
            self.wandb_logger.after_val_epoch()

    # iter
    def before_train_iter(self):
        if self.wandb_logger:
            self.wandb_logger.before_train_iter()

    def after_train_iter(self):
        if self.wandb_logger:
            self.wandb_logger.after_train_iter()

    def before_val_iter(self):
        if self.wandb_logger:
            self.wandb_logger.before_val_iter()

    def after_val_iter(self):
        if self.wandb_logger:
            self.wandb_logger.after_val_iter()

