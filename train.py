import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra

from dataloader import default_collator, load, to_tfdataset
from metrics import accuracy, loss
from models.MainModels import BinaryClassificationModel
from trainer import TrainArgument, Trainer


@hydra.main(config_name="config.yml")
def main(cfg):
    train_dataset, eval_dataset = load(**cfg.DATASETS)

    train_dataset = to_tfdataset(train_dataset, cfg.TRAINARGS.signature)
    eval_dataset = (
        to_tfdataset(eval_dataset, cfg.TRAINARGS.signature)
        if eval_dataset is not None
        else None
    )

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
