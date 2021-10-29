import os
from datetime import datetime as dt

import tensorflow as tf
from pytz import timezone


class TrainArgument:
    def __init__(self, **kwargs):
        # training parameters
        self.strategy = self.get_strategy(kwargs.get("use_gpu", True))
        self.train_batch_size = kwargs.get("train_batch_size", 4)
        self.train_global_batch_size = (
            self.train_batch_size * self.strategy.num_replicas_in_sync
        )
        self.eval_batch_size = kwargs.get("eval_batch_size", 4)
        self.eval_global_batch_size = (
            self.eval_batch_size * self.strategy.num_replicas_in_sync
        )
        self.epochs = kwargs.get("epochs", 1)
        self.eval_epoch = kwargs.get("eval_epoch", self.epochs)
        self.eval_epoch = self.epochs if self.eval_epoch == -1 else self.eval_epoch

        # checkpoint
        self.checkpoint_dir = kwargs.get("checkpoint_dir")
        self.save_epoch = kwargs.get("save_epoch", 1)
        self.save_total_limit = kwargs.get("save_total_limit", int(1e9))
        if self.checkpoint_dir is None:
            self.checkpoint_dir = "./ckpt"
            self.save_total_limit = 1

        # logging
        self.logging_dir = kwargs.get("logging_dir")
        if self.logging_dir is not None:
            self.logging_dir = os.path.join(
                self.logging_dir,
                dt.now(timezone("Asia/Seoul")).strftime("%Y%m%d%H%M%S"),
            )
        self.logging_steps = kwargs.get("logging_steps", 100)
        self.logging_print = kwargs.get("logging_print", False)

        # optimizer
        self.learning_rate = kwargs.get("learning_rate", 5e-05)
        self.warmup_steps = kwargs.get("warmup_steps", 0)
        self.adam_beta1 = kwargs.get("adam_beta1", 0.9)
        self.adam_beta2 = kwargs.get("adam_beta2", 0.98)
        self.adam_epsilon = kwargs.get("adam_epsilon", 1e-9)
        self.power = kwargs.get("power", 1.0)

    def get_strategy(self, use_gpu):
        gpus = tf.config.list_physical_devices("GPU")

        if use_gpu:
            if len(gpus) == 0:
                strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
            elif len(gpus) == 1:
                strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            elif len(gpus) > 1:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.ReductionToOneDevice()
                )
        else:
            strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

        return strategy
