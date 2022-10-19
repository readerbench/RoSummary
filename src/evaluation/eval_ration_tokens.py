from __future__ import annotations

import json
from typing import Tuple, List, Any

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from statistics import mean

from scipy.stats import spearmanr, pearsonr
from transformers import GPT2Tokenizer
from sklearn.metrics import mean_squared_error, mean_absolute_error

tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')


def _worker(items: list[dict]) -> Tuple[List[Any], List[Any]]:
    ration_tokens_original = list()
    ration_tokens_generate = list()

    for item in tqdm(items):
        if int(len(item["context"]) * 0.9) < len(item['original']):
            continue

        token_contex = tokenizer.encode(item['context'])
        token_generate = tokenizer.encode(item['generate'])

        ration_tokens_original.append(item['ration-tokens'])
        ration_tokens_generate.append(len(token_generate) / len(token_contex))

    return ration_tokens_original, ration_tokens_generate


def eval_no_sentences_items(items: dict) -> dict:
    ration_tokens_original = []
    ration_tokens_generate = []

    no_workers = 100
    batch_size = len(items) // no_workers + 1
    batches = [items[i * batch_size: (i + 1) * batch_size] for i in range(no_workers)]

    with Pool(no_workers) as pool:
        results = pool.map_async(_worker, batches)
        pool.close()
        pool.join()

        for result in results.get():
            ration_tokens_original.extend(result[0])
            ration_tokens_generate.extend(result[1])

    return {
        "mse": str(mean_squared_error(ration_tokens_original, ration_tokens_generate)),
        "mae": str(mean_absolute_error(ration_tokens_original, ration_tokens_generate)),
        "spearman": spearmanr(ration_tokens_generate, ration_tokens_original),
        "pearson": pearsonr(ration_tokens_generate, ration_tokens_original)
    }


def eval_all_generate_no_sentences(path_director: str) -> None:
    results = dict()

    for path_director in tqdm(Path(path_director).glob("**/sample_output/")):
        results_director = dict()

        for path in tqdm(Path(path_director).glob("*.json")):

            if 'top-k' in path.name:
                continue

            with path.open('r') as file_input:
                data = json.load(file_input)

            if 'ration-tokens' not in data[0]:
                continue

            results_director[path.name.replace('.json', '')] = eval_no_sentences_items(data)
        if not results_director:
            continue

        results[path_director.parent.name] = results_director

    with open('result_no_sentences.json', 'w+') as file_output:
        json.dump(results, file_output, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    eval_all_generate_no_sentences("../../genaration")
