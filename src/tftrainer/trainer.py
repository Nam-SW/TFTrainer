from math import ceil
from typing import Callable, Dict, List, Optional, Union

import tensorflow as tf
from tqdm import tqdm

from .optimizer import create_optimizer
from .trainarguments import TrainArgument


class Trainer:
    def __init__(
        self,
        model: tf.keras.Model,
        args: TrainArgument,
        train_dataset: tf.data.Dataset,
        loss_function: Callable,
        eval_dataset: Optional[tf.data.Dataset] = None,
        data_collator: Optional[Callable] = None,
        log_function: Optional[Callable] = None,
        optimizers: Optional[List] = [None, None],
        metrics: Optional[Union[List[Callable], Callable]] = None,
    ):
        self.model = model
        self.args = args
        self.data_collator = data_collator
        self.loss_function = loss_function

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.do_eval = self.eval_dataset is not None

        self.log_function = log_function
        self.optimizer, self.lr_scheduler = optimizers
        self.set_tensorboard(self.args.logging_dir)

        self.set_metrics(metrics)

    def set_metrics(self, metrics: Optional[Union[List[Callable], Callable]] = None):
        self.loss = tf.keras.metrics.Mean(name="loss")

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
            self.metrics = None

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

    def set_tensorboard(self, logging_dir: Optional[str] = None):
        self.logging = logging_dir is not None
        if self.logging:
            self.logger = tf.summary.create_file_writer(logging_dir)

    def set_optimizer(
        self,
        num_training_steps: int,
        optimizer: Optional[tf.keras.optimizers.Optimizer] = None,
        lr_scheduler: Optional[Callable[[int], float]] = None,
    ):
        if optimizer is None:
            self.optimizer, self.lr_scheduler = create_optimizer(
                init_lr=self.args.learning_rate,
                num_train_steps=num_training_steps,
                num_warmup_steps=self.args.warmup_steps,
                min_lr_ratio=self.args.min_lr_ratio,
                adam_beta1=self.args.adam_beta1,
                adam_beta2=self.args.adam_beta2,
                adam_epsilon=self.args.adam_epsilon,
                power=self.args.power,
            )
        else:
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

    def get_dataset(
        self,
        dataset: tf.data.Dataset,
        batch_size: int,
        data_length: int = None,
    ):
        dataset = dataset.batch(batch_size)
        if self.data_collator is not None:
            dataset = dataset.map(self.data_collator).prefetch(
                tf.data.experimental.AUTOTUNE
            )

        if data_length is None:
            length = len(dataset)
        else:
            length = ceil(data_length / batch_size)

        return self.args.strategy.experimental_distribute_dataset(dataset), length

    def get_log(self, tag="", epoch=None, lr=None):
        log_dict = dict()
        if tag and not tag.startswith("/"):
            tag = "/" + tag

        if epoch:
            log_dict["epoch" + tag] = epoch
        if lr:
            log_dict["lr" + tag] = lr

        log_dict["loss" + tag] = self.loss.result()

        if self.metrics_func is not None:
            for m in self.metrics:
                log_dict[m.name + tag] = m.result()

        return log_dict

    def log(self, log_dict: Dict[str, float], step: Union[int, float]):
        self.logger.flush()
        with self.logger.as_default():
            for name, value in log_dict.items():
                tf.summary.scalar(name, value, step=step)

        if self.log_function is not None:
            self.log_function(log_dict)

    @tf.function
    def step(self, x, y, training=False):
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

    def train(
        self,
        dataset: Optional[tf.data.Dataset] = None,
        data_length: Optional[int] = None,
        callbacks=None,
    ):
        dataset, step_per_epoch = self.get_dataset(
            self.train_dataset if dataset is None else dataset,
            batch_size=self.args.train_global_batch_size,
            data_length=data_length,
        )
        num_training_step = step_per_epoch * self.args.epochs

        with self.args.strategy.scope():
            if self.optimizer is None:
                self.set_optimizer(num_training_step, self.optimizer, self.lr_scheduler)

            self.set_checkpoint()

            callbacks = tf.keras.callbacks.CallbackList(
                callbacks,
                add_history=True,
                add_progbar=False,
                model=self.model,
                verbose=0,
                epochs=self.args.epochs,
                steps=step_per_epoch,
            )
            callbacks.on_train_begin()

            pbar = tqdm(total=num_training_step)
            pbar.update(self.ckpt.step.numpy())

            for epoch in range(
                self.ckpt.step.numpy() // step_per_epoch, self.args.epochs
            ):
                self.loss.reset_states()
                if self.metrics_func is not None:
                    for m in self.metrics:
                        m.reset_states()

                callbacks.on_epoch_begin(epoch)

                for x, y in dataset:
                    callbacks.on_train_batch_begin(self.ckpt.step.numpy())
                    self.distributed_step(x, y, training=True)

                    log_dict = self.get_log(
                        tag="train",
                        epoch=self.ckpt.step.numpy() / step_per_epoch,
                        lr=self.lr_scheduler(self.ckpt.step).numpy()
                        if self.lr_scheduler is not None
                        else None,
                    )

                    if (
                        self.logging
                        and self.ckpt.step.numpy() % self.args.logging_steps == 0
                    ):
                        self.log(log_dict, self.ckpt.step.numpy())

                        if self.args.logging_print:
                            str_log_dict = "train step {}: {}".format(
                                self.ckpt.step.numpy(),
                                ", ".join(
                                    [f"{k}: {v: .4f}" for k, v in log_dict.items()]
                                ),
                            )
                            print(str_log_dict)

                        callbacks.on_train_batch_end(self.ckpt.step.numpy(), log_dict)

                    self.ckpt.step.assign_add(1)
                    pbar.update(1)

                callbacks.on_epoch_end(epoch, log_dict)

                if (epoch + 1) % self.args.save_epoch == 0:
                    self.save_checkpoint()

                if self.do_eval and (epoch + 1) % self.args.eval_epoch == 0:
                    self.eval(view_progress=False)

            pbar.close()
            callbacks.on_train_end(logs=log_dict)

    def eval(
        self,
        dataset: Optional[tf.data.Dataset] = None,
        data_length: Optional[int] = None,
        view_progress: Optional[bool] = True,
    ):
        dataset, step_per_epoch = self.get_dataset(
            self.eval_dataset if dataset is None else dataset,
            batch_size=self.args.eval_global_batch_size,
            data_length=data_length,
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

        log_dict = self.get_log(tag="eval")
        if self.logging:
            self.log(log_dict, self.ckpt.step.numpy())

        return log_dict
