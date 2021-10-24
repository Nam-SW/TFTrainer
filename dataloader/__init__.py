import random
from functools import partial
from multiprocessing import Pool
from os.path import abspath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from dataloader.utils import (
    check_valid_jsonl,
    dump_jsonl,
    json_to_jsonl,
    jsonl_to_json,
    load_jsonl,
)


class DataLoader:
    def __init__(
        self,
        data_path: Optional[str] = None,
        data: Optional[Union[List[Dict], Dict[str, List]]] = None,
    ) -> None:
        if data_path is None and data is None:
            raise ValueError("Either data_path or data must be given.")
        if data_path is not None and data is not None:
            raise ValueError("Only one of data_path and data must be given.")

        if data_path is not None:
            data_path = [data_path] if isinstance(data_path, str) else data_path
            self.data = sum([load_jsonl(p) for p in data_path], [])

        if data is not None:
            check_valid_jsonl(data)
            if isinstance(data, dict):
                data = json_to_jsonl(data)

            self.data = data

    def __getitem__(self, key: Union[str, int, slice, List[str]]) -> Any:
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return jsonl_to_json(self.data[key])
        elif isinstance(key, str):
            return [row[key] for row in self.data]
        elif isinstance(key, list) and not (False in [isinstance(k, str) for k in key]):
            return {k: self[k] for k in key}
        else:
            raise KeyError("`key` argument must be either str or int.")

    def __setitem__(self, key: str, value: List[Any]) -> Any:
        if not isinstance(key, str):
            raise KeyError("`key` argument must be str.")
        if len(value) != len(self.data):
            raise ValueError(
                "Length of value ({}) does not match length of index ({})".format(
                    len(value), len(self.data)
                )
            )

        for i in range(len(self.data)):
            self.data[i][key] = value[i]

    def __delitem__(self, key: str) -> Any:
        if key not in self.keys():
            raise KeyError(key)

    def __len__(self):
        return len(self.data)

    def __np_to_list(self, sample: Dict) -> Dict:
        return {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sample.items()
        }

    def keys(self) -> List[str]:
        return list(self.data[0].keys())

    def shuffle(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            random.seed(seed)

        random.shuffle(self.data)

    def apply(
        self,
        func: Callable,
        share_values: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        worker: int = 1,
    ) -> None:
        data = (
            self.data
            if batch_size is None
            else [
                jsonl_to_json(self.data[i : i + batch_size])
                for i in range(0, len(self.data), batch_size)
            ]
        )

        result = []
        if worker == 1:
            for batch in tqdm(data):
                r = self.__np_to_list(func(*batch))

                if batch_size is None:
                    result.append(r)
                else:
                    result += json_to_jsonl(r)
        else:
            with Pool(worker) as pool:
                for r in tqdm(
                    pool.imap(
                        func=partial(func, share_values=share_values),
                        iterable=data,
                    ),
                    total=len(data),
                ):
                    r = self.__np_to_list(r)
                    if batch_size is None:
                        result.append(r)
                    else:
                        result += json_to_jsonl(r)

        self.data = result

    def train_test_split(test_size: float = 0.2) -> Tuple[Dict, Dict]:
        pass

    def save(self, path: str) -> None:
        dump_jsonl(self.data, path)


def load(
    data_path: str,
    share_values: Optional[Dict[str, Any]] = None,
    map_args: Optional[List[Dict[str, Any]]] = None,
    shuffle_seed: Optional[int] = None,
    train_test_split: Optional[float] = None,
):

    data = DataLoader(abspath(data_path))

    if map_args is not None:
        for args in map_args:
            data.apply(share_values=share_values, **args)

    if shuffle_seed is not None:
        data.shuffle(seed=shuffle_seed)

    if train_test_split is not None:
        pass

    return data
