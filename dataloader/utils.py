import json
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import tensorflow as tf
from numpy.lib.arraysetops import isin


def dump_jsonl(data: List[Dict], output_path: str, append: bool = False) -> None:
    mode = "a+" if append else "w"
    with open(output_path, mode, encoding="utf-8") as f:
        for line in data:
            json_record = json.dumps(line, ensure_ascii=False)
            f.write(json_record + "\n")


def load_jsonl(input_path: str) -> list:
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.rstrip("\n|\r")))
    return data


def jsonl_to_json(data: List[Dict]) -> Dict:
    return {k: [line[k] for line in data] for k in data[0].keys()}


def json_to_jsonl(data: Dict) -> List[Dict]:
    keys = list(data.keys())
    return [{k: data[k][i] for k in keys} for i in range(len(data[keys[0]]))]


def check_valid_jsonl(data: Union[Dict, List]) -> None:
    if isinstance(data, list):
        keys = set(data[0].keys())
        for row in data:
            if not isinstance(row, dict):
                raise ValueError("All values in the list must be dictionary.")
            elif set(row.keys()) != keys:
                raise ValueError("All values in the list must have the same key.")

    elif isinstance(data, dict):
        len_list = [len(v) for v in data.values()]
        if sum(len_list) / len(len_list) != len_list[0]:
            raise ValueError("All values in the dictionary must have the same length.")

    else:
        raise ValueError("json or jsonl object must be a dictionary or list value.")


def convert_tf_datasets(
    dataset,
    input_key: Union[str, List[str], Tuple[str]],
    labels_key: Union[str, List[str], Tuple[str]],
    type: Literal["dict", "tuple"] = "dict",
    batch_size: int = 1,
    collator: Optional[Callable] = None,
) -> tf.data.Dataset:
    if type == "dict":
        x = dataset[input_key]
        y = dataset[labels_key]
    elif type == "tuple":
        x = (
            tuple([dataset[key] for key in input_key])
            if isinstance(input_key, (list, tuple))
            else dataset[input_key]
        )
        y = (
            tuple([dataset[key] for key in labels_key])
            if isinstance(labels_key, (list, tuple))
            else dataset[labels_key]
        )
        # y = tuple([dataset[key] for key in labels_key])

    # TODO: 여기 오래걸림
    tf_dataset = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

    if collator is not None:
        tf_dataset = tf_dataset.map(collator).prefetch(tf.data.experimental.AUTOTUNE)

    return tf_dataset
