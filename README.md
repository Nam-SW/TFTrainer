# TFTrainer
 tensorflow에서 모델 학습 및 체크포인트, tensorboard 로깅까지 한번에 처리해주는 trainer.

## install
```bash
$ pip install TFTrainer
```

## Usage
```py
from tftrainer import TrainArgument, Trainer


args = TrainArgument(
    use_gpu=True,
    train_batch_size=64,
    eval_batch_size=64,
    epochs=30,
    checkpoint_dir="ckpt",
    save_epoch=3,
    save_total_limit=10,
    logging_dir="logs",
    logging_steps=50,
    learning_rate=0.0001,
)

train_dataset = [tensorflow dataset]
eval_dataset = [tensorflow dataset or None]

with args.strategy.scope():
    model = [TF2 model]

trainer = Trainer(
    model,
    args,
    train_dataset,
    eval_dataset=eval_dataset,
    loss_function="categorical_crossentropy",
    metrics=["accuracy"],
    callbacks=[tf callbacks or None],
)

trainer.train()
```