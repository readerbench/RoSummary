from __future__ import annotations

import json
from typing import Tuple, List, Any

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from statistics import mean

import spacy
from sklearn.metrics import mean_squared_error, mean_absolute_error

model_ro = spacy.load('ro_core_news_lg')


def _worker(items: list[dict]) -> Tuple[List[Any], List[int]]:
    no_words_original = list()
    no_words_generate = list()

    for item in tqdm(items):
        if int(len(item["context"]) * 0.9) < len(item['original']):
            continue
        sentences = [sentence.text for sentence in model_ro(item['generate']).sents]

        no_words_original.append(item['no-sentences'])
        no_words_generate.append(len(sentences))

    return no_words_original, no_words_generate


def eval_no_sentences_items(items: dict) -> dict:
    no_words_original = []
    no_words_generate = []

    no_workers = 20
    batch_size = len(items) // no_workers + 1
    batches = [items[i * batch_size: (i + 1) * batch_size] for i in range(no_workers)]

    with Pool(no_workers) as pool:
        results = pool.map_async(_worker, batches)
        pool.close()
        pool.join()

        for result in results.get():
            no_words_original.extend(result[0])
            no_words_generate.extend(result[1])

    return {
        "mse": str(mean_squared_error(no_words_original, no_words_generate)),
        "mae": str(mean_absolute_error(no_words_original, no_words_generate)),
        "mean-original": mean(no_words_original),
        "mean-generate": mean(no_words_generate)
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

            if 'no-sentences' not in data[0]:
                continue

            results_director[path.name.replace('.json', '')] = eval_no_sentences_items(data)
        if not results_director:
            continue

        results[path_director.parent.name] = results_director

    with open('result_no_sentences.json', 'w+') as file_output:
        json.dump(results, file_output, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    eval_all_generate_no_sentences("../../genaration")
