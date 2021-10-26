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

        # checkpoint
        self.checkpoint_dir = kwargs.get("checkpoint_dir")
        self.save_epoch = kwargs.get("save_epoch", 1)
        self.save_total_limit = kwargs.get("save_total_limit", int(1e9))

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

    def set_checkpoint(self, checkpoint_dir=None, step_per_epoch=None):
        if checkpoint_dir is None:
            self.checkpoint = False
            return

        self.checkpoint = True
        if not os.path.isdir(checkpoint_dir):
            os.mkdir(checkpoint_dir)

        if step_per_epoch is not None and os.listdir(checkpoint_dir):
            last_ckpt = sorted(os.listdir(checkpoint_dir))[-1]
            with open(
                os.path.join(checkpoint_dir, last_ckpt, "ckpt_info.json"),
                "r",
                encoding="utf-8",
            ) as f:
                info = json.load(f)
                self.global_step = info["step"]
                self.ckpt_step = info["step"]
                self.lr = info["lr"]
                self.epoch = info["epoch"]

            with self.args.strategy.scope():
                self.model = self.model.load(os.path.join(checkpoint_dir, last_ckpt))
                print("load checkpoint at " + last_ckpt)

    def save_checkpoint(self):
        assert self.checkpoint
        epoch_str = str(self.epoch + 1).zfill(len(str(self.args.epochs)))

        ckpt_list = os.listdir(self.args.checkpoint_dir)
        save_dir = os.path.join(self.args.checkpoint_dir, "epoch_" + epoch_str)
        if self.args.logging_print:
            print("saved at " + save_dir)

        if self.args.save_total_limit <= len(ckpt_list):
            first_one = os.path.join(self.args.checkpoint_dir, sorted(ckpt_list)[0])
            rmtree(first_one)

        self.model.save(save_dir)
        with open(os.path.join(save_dir, "ckpt_info.json"), "w", encoding="utf-8") as f:
            info = {
                "step": self.global_step,
                "lr": float(
                    self.lr_scheduler(self.global_step - self.ckpt_step).numpy()
                ),
                "epoch": self.epoch + 1,
            }
            json.dump(info, f)

    def set_tensorboard(self, logging_dir=None):
        if logging_dir is None:
            self.logging = False
            return

        self.logging = True

        self.logger = tf.summary.create_file_writer(logging_dir)

    def set_optimizer(self, optimizer=None, lr_scheduler=None):
        num_training_steps = (
            ceil(len(self.train_dataset) / self.args.train_global_batch_size)
            * self.args.epochs
        )

        if optimizer is None:
            self.optimizer, self.lr_scheduler = create_optimizer(
                self.lr if self.lr > 0.0 else self.args.learning_rate,
                num_training_steps - self.global_step,
                max(0, self.args.warmup_steps - self.global_step),
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
        pred = self.model(**x, training=training)
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

        self.set_checkpoint(self.args.checkpoint_dir)

        pbar = tqdm(total=step_per_epoch * self.args.epochs)
        pbar.update(self.global_step)

        with self.args.strategy.scope():
            if self.optimizer is None:
                self.set_optimizer(self.optimizer, self.lr_scheduler)

            for epoch in range(self.args.epochs):
                self.now_epoch = epoch
                self.loss.reset_states()
                if self.metrics_func is not None:
                    for m in self.metrics:
                        m.reset_states()

                for step, (x, y) in enumerate(dataset):
                    global_step = step_per_epoch * epoch + step

                    self.distributed_step(x, y, training=True)

                    if self.logging and global_step % self.args.logging_steps == 0:
                        log_dict = dict()
                        tag = "/train"

                        log_dict["epoch" + tag] = global_step / step_per_epoch
                        if self.lr_scheduler is not None:
                            lr = self.lr_scheduler(global_step).numpy()
                            log_dict["lr" + tag] = lr

                        log_dict["loss" + tag] = self.loss.result()

                        if self.metrics_func is not None:
                            for m in self.metrics:
                                log_dict[m.name + tag] = m.result()

                        self.log(log_dict, global_step)

                        if self.args.logging_print:
                            str_log_dict = "train step {}: {}".format(
                                global_step,
                                ", ".join(
                                    [f"{k}: {v: .4f}" for k, v in log_dict.items()]
                                ),
                            )
                            print(str_log_dict)

                    pbar.update(1)

                if self.checkpoint and (epoch + 1) % self.args.save_epoch == 0:
                    self.save_checkpoint()

                if self.do_eval and (epoch + 1) % self.args.eval_epoch == 0:
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
            self.log(log_dict, self.epoch)

        return log_dict
