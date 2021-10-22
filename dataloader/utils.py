import json
from typing import Dict, List


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


def default_collator(data):
    return data
