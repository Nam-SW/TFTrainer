import json
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import tensorflow as tf


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
        raise ValueError("json or jsonl argument must be a dictionary or list value.")


def get_signature(data: List) -> Dict[str, tf.TensorSpec]:
    tf_sample = tf.constant(data[0])
    return tf.TensorSpec(shape=tf_sample.shape, dtype=tf_sample.dtype)


def convert_tf_datasets(
    dataset,
    input_key: Union[str, List[str], Tuple[str]],
    labels_key: Union[str, List[str], Tuple[str]],
    dtype: Literal["dict", "tuple"] = "tuple",
) -> tf.data.Dataset:
    if dtype not in ["dict", "tuple"]:
        raise ValueError("dtype argument must be a 'dict' or 'tuple'.")

    x = dataset[input_key]
    y = dataset[labels_key]

    if isinstance(input_key, str):
        x_signature = get_signature(x)
    elif dtype == "dict":
        x_signature = {key: get_signature(dataset[key]) for key in input_key}
    elif dtype == "tuple":
        x_signature = tuple([get_signature(dataset[key]) for key in input_key])

    if isinstance(labels_key, str):
        y_signature = get_signature(y)
    elif dtype == "dict":
        y_signature = {key: get_signature(dataset[key]) for key in labels_key}
    elif dtype == "tuple":
        y_signature = tuple([get_signature(dataset[key]) for key in labels_key])

    def _generate():
        for batch_x, batch_y in zip(x, y):
            if dtype == "tuple":
                if not isinstance(input_key, str):
                    batch_x = tuple(batch_x.values())
                if not isinstance(labels_key, str):
                    batch_y = tuple(batch_y.values())

            yield batch_x, batch_y

    tf_dataset = tf.data.Dataset.from_generator(
        _generate, output_signature=(x_signature, y_signature)
    )
    tf_dataset = tf_dataset.apply(tf.data.experimental.assert_cardinality(len(x)))

    return tf_dataset
