import json
import os
from datetime import datetime as dt
from math import ceil
from shutil import rmtree

import tensorflow as tf
from pytz import timezone
from tqdm import tqdm

from trainer.optimizer import create_optimizer

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


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


class Trainer:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        loss_function,
        eval_dataset=None,
        data_collator=None,
        optimizers=[None, None],
        metrics=None,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.loss_function = loss_function

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.do_eval = self.eval_dataset is not None

        self.optimizer, self.lr_scheduler = optimizers
        self.set_tensorboard(self.args.logging_dir)

        self.set_metrics(metrics)

    def set_metrics(self, metrics=None):
        self.loss = tf.keras.metrics.Mean(name="loss")
        self.global_step = 0
        self.ckpt_step = 0
        self.lr = 0.0
        self.epoch = 0

        metrics = [] if metrics is None else metrics
        metrics = [metrics] if hasattr(metrics, "__call__") else metrics
        for i in range(len(metrics)):
            if not hasattr(metrics[i], "__name__"):
                metrics[i].__name__ = metrics[i].__class__.__name__

        if isinstance(metrics, list) or isinstance(metrics, tuple):
            self.metrics_func = metrics
            self.metrics = [
                tf.keras.metrics.Mean(name=m.__name__) for m in self.metrics_func
            ]
        else:
            self.metrics_func = None
            self.train_metrics = None
            self.eval_metrics = None

    # TODO: model save, load 통일

    def set_checkpoint(self):
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(1), optimizer=self.optimizer, net=self.model
        )
        self.ckpt_manager = tf.train.CheckpointManager(
            self.ckpt, self.args.checkpoint_dir, max_to_keep=self.args.save_total_limit
        )

        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            print("load checkpoint from " + self.ckpt_manager.latest_checkpoint)

    def save_checkpoint(self):
        save_path = self.ckpt_manager.save()
        return save_path

    def set_tensorboard(self, logging_dir=None):
        if logging_dir is None:
            self.logging = False
            return

        self.logging = True

        self.logger = tf.summary.create_file_writer(logging_dir)

    def set_optimizer(self, num_training_steps, optimizer=None, lr_scheduler=None):
        if optimizer is None:
            self.optimizer, self.lr_scheduler = create_optimizer(
                self.args.learning_rate,
                num_training_steps,
                self.args.warmup_steps,
                adam_beta1=self.args.adam_beta1,
                adam_beta2=self.args.adam_beta2,
                adam_epsilon=self.args.adam_epsilon,
                power=self.args.power,
            )
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

    def get_dataset(self, dataset, batch_size):
        dataset = dataset.batch(batch_size)
        if self.data_collator is not None:
            dataset = dataset.map(self.data_collator).prefetch(
                tf.data.experimental.AUTOTUNE
            )

        return self.args.strategy.experimental_distribute_dataset(dataset), len(dataset)

    def log(self, log_dict, step):
        self.logger.flush()
        with self.logger.as_default():
            for name, value in log_dict.items():
                tf.summary.scalar(name, value, step=step)

    @tf.function
    def step(self, x, y, training=False):
        if isinstance(x, dict):
            pred = self.model(**x, training=training)
        elif isinstance(x, tuple):
            pred = self.model(x, training=training)

        loss = self.loss_function(y, pred)
        if self.metrics_func is not None:
            metrics = [m(y, pred) for m in self.metrics_func]

        if training:
            gradients = tf.gradients(loss, self.model.trainable_variables)
            gradients = [
                g if g is not None else tf.zeros_like(v)
                for g, v in zip(gradients, self.model.trainable_variables)
            ]
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )

        self.loss(loss)
        if self.metrics_func is not None:
            for i, m in enumerate(metrics):
                self.metrics[i](m)

    @tf.function
    def distributed_step(self, x, y, training=False):
        self.args.strategy.run(self.step, args=(x, y, training))

    def train(self, dataset=None):
        dataset, step_per_epoch = self.get_dataset(
            self.train_dataset if dataset is None else dataset,
            batch_size=self.args.train_global_batch_size,
        )
        num_training_step = step_per_epoch * self.args.epochs

        with self.args.strategy.scope():
            if self.optimizer is None:
                self.set_optimizer(num_training_step, self.optimizer, self.lr_scheduler)

            self.set_checkpoint()

            pbar = tqdm(total=num_training_step)
            pbar.update(self.ckpt.step.numpy())

            for epoch in range(
                self.ckpt.step.numpy() // step_per_epoch, self.args.epochs
            ):
                self.loss.reset_states()
                if self.metrics_func is not None:
                    for m in self.metrics:
                        m.reset_states()

                for step, (x, y) in enumerate(dataset):

                    self.distributed_step(x, y, training=True)

                    if (
                        self.logging
                        and self.ckpt.step.numpy() % self.args.logging_steps == 0
                    ):
                        log_dict = dict()
                        tag = "/train"

                        log_dict["epoch" + tag] = (
                            self.ckpt.step.numpy() / step_per_epoch
                        )
                        if self.lr_scheduler is not None:
                            lr = self.lr_scheduler(self.ckpt.step).numpy()
                            log_dict["lr" + tag] = lr

                        log_dict["loss" + tag] = self.loss.result()

                        if self.metrics_func is not None:
                            for m in self.metrics:
                                log_dict[m.name + tag] = m.result()

                        self.log(log_dict, self.ckpt.step.numpy())

                        if self.args.logging_print:
                            str_log_dict = "train step {}: {}".format(
                                self.ckpt.step.numpy(),
                                ", ".join(
                                    [f"{k}: {v: .4f}" for k, v in log_dict.items()]
                                ),
                            )
                            print(str_log_dict)

                    self.ckpt.step.assign_add(1)
                    pbar.update(1)

                if epoch % self.args.save_epoch == 0:
                    self.save_checkpoint()

                if self.do_eval and epoch % self.args.eval_epoch == 0:
                    self.eval(view_progress=False)

    def eval(self, dataset=None, view_progress=True):
        dataset, step_per_epoch = self.get_dataset(
            self.eval_dataset if dataset is None else dataset,
            batch_size=self.args.eval_global_batch_size,
        )

        if view_progress:
            pbar = tqdm(total=step_per_epoch)

        with self.args.strategy.scope():
            self.loss.reset_states()
            if self.metrics_func is not None:
                for m in self.metrics:
                    m.reset_states()

            for (x, y) in dataset:
                self.distributed_step(x, y, training=False)

                if view_progress:
                    pbar.update(1)

        tag = "eval/"
        log_dict = {tag + "loss": self.loss.result()}
        if self.metrics_func is not None:
            for m in self.metrics:
                log_dict[tag + m.name] = m.result()

        if self.logging:
            self.log(log_dict, self.ckpt.step.numpy())

        return log_dict
