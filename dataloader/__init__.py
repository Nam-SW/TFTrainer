import random
from functools import partial
from multiprocessing import Pool
from os.path import abspath
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from tqdm import tqdm

from dataloader.utils import dump_jsonl, json_to_jsonl, jsonl_to_json, load_jsonl

__all__ = ["user_define", "utils"]


class DataLoader:
    def __init__(
        self,
        data_path: str,
    ) -> None:
        data_path = [data_path] if isinstance(data_path, str) else data_path
        self.data = sum([load_jsonl(p) for p in data_path], [])

    def __getitem__(self, key: Union[str, int, slice]) -> Any:
        if isinstance(key, int):
            return self.data[key]
        elif isinstance(key, slice):
            return jsonl_to_json(self.data[key])
        elif isinstance(key, str):
            return [row[key] for row in self.data]
        else:
            raise KeyError("`key` object must be either str or int.")

    def __np_to_list(self, sample: Dict) -> Dict:
        return {
            k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in sample.items()
        }

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
