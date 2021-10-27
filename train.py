import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import hydra

from dataloader import load, user_define
from metrics import loss
from models.MainModels import Transformer
from trainer import TrainArgument, Trainer


@hydra.main(config_name="config.yml")
def main(cfg):
    # step 1. set train arguments. This process should be done first.
    train_args = TrainArgument(**cfg.TRAINARGS)

    # step 2. load datasets.
    with open(cfg.ETC.tokenizer_path, "r", encoding="utf-8") as f:
        tokenizer = json.load(f)

    share_dict = dict(tokenizer=tokenizer, **cfg.DATASETS.share_dict)
    map_args = [
        dict(func=func, **arg)
        for func, arg in zip(user_define.aplly_list, cfg.DATASETS.map_args_list)
    ]
    train_dataset, eval_dataset = load(
        data_path=cfg.DATASETS.data_path,
        # data_path=pd.read_csv(cfg.DATASETS.data_path).to_dict("list"),  # if data_path == "data_sample.csv"
        input_key=cfg.DATASETS.input_key,
        labels_key=cfg.DATASETS.labels_key,
        share_values=share_dict,
        map_args=map_args,
        shuffle_seed=cfg.DATASETS.shuffle_seed,
        train_test_split=cfg.DATASETS.train_test_split,
        dtype=cfg.DATASETS.dtype,
    )

    # step 3. set model.
    with train_args.strategy.scope():
        model = Transformer(vocab_size=len(tokenizer), **cfg.MODEL)

    # step 4. set trainer.
    trainer = Trainer(
        model,
        train_args,
        train_dataset,
        eval_dataset=eval_dataset,
        loss_function=loss,
        data_collator=user_define.data_collator,
    )

    # step 5. train and evaluation.
    trainer.train()

    # step 6. save model.
    model.save(cfg.ETC.output_dir)


if __name__ == "__main__":
    main()
