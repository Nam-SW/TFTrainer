from os.path import abspath, splitext
from typing import Optional

from datasets import load_dataset, logging

logging.set_verbosity(logging.ERROR)


# Write code to load custom data.
def load(
    train_data_path: str,
    eval_data_path: Optional[str] = None,
    train_test_split: Optional[float] = None,
    worker: int = 1,
    batch_size: int = 1000,
    shuffle_seed: Optional[int] = None,
):
    def _wraping(data):
        data["labels"] = [data["labels"]]
        return data

    train_data_path = abspath(train_data_path)
    is_eval = False
    _, extention = splitext(train_data_path)

    datafiles = {"train": train_data_path}
    if eval_data_path is not None:
        assert (
            train_test_split is None
        ), "Only one of eval_data_path and train_test_split must be entered."
        datafiles["test"] = abspath(eval_data_path)
        is_eval = True

    if train_test_split is not None:
        assert (
            0.0 < train_test_split < 1.0
        ), "train_test_split must be a value between 0 and 1"
        train_test_split = int(train_test_split * 100)
        train_test_split = {
            "train": f"train[:{train_test_split}%]",
            "test": f"train[{train_test_split}%:]",
        }
        is_eval = True

    data = load_dataset(
        extention.replace(".", ""), data_files=datafiles, split=train_test_split
    )
    if shuffle_seed:
        data = data.shuffle(seed=shuffle_seed)

    data = data.map(
        _wraping,
        # batched=True,
        # batch_size=batch_size,
        num_proc=worker,
    )

    return data["train"], (data["test"] if is_eval else None)


# Write preprocessor code to run in batches.
def default_collator(data):
    return data
