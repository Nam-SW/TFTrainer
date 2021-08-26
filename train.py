import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra

from dataloader import default_collator, load
from models.MainModels import BinaryClassificationModel
from nn import accuracy, loss
from trainer import TrainArgument, Trainer


@hydra.main(config_name="config.yml")
def main(cfg):
    train_dataset, eval_dataset = load(**cfg.DATASETS)

    args = TrainArgument(**cfg.TRAINARGS)

    with args.strategy.scope():
        model = BinaryClassificationModel(**cfg.MODEL)

    trainer = Trainer(
        model,
        args,
        train_dataset,
        loss,
        eval_dataset=eval_dataset,
        data_collator=default_collator,
        metrics=accuracy,
    )

    trainer.train()

    model.save(cfg.ETC.output_dir)


if __name__ == "__main__":
    main()
