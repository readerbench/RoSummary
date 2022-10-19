from __future__ import annotations

import json
from typing import Tuple, List, Any

from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

import spacy
from nltk import ngrams
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error

model_ro = spacy.load('ro_core_news_lg')


def _worker(items: list[dict]) -> Tuple[List[Any], List[Any]]:
    overlap_original = list()
    overlap_generate = list()

    for item in tqdm(items):
        if int(len(item["context"]) * 0.9) < len(item['original']):
            continue

        nlp_summary = [token.text for token in model_ro(item['generate']) if not token.is_stop or not token.is_punct]
        nlp_news = [token.text for token in model_ro(item["context"]) if not token.is_stop or not token.is_punct]

        ng_4_summary = set(list(ngrams(nlp_summary, 4)))
        ng_4_news = set(list(ngrams(nlp_news, 4)))
        count_overlap = 0

        for gram in ng_4_summary:
            if gram in ng_4_news:
                count_overlap += 1

        overlap_original.append(item['overlap'])
        overlap_generate.append(count_overlap / len(nlp_summary))

    return overlap_original, overlap_generate


def eval_items(items: dict) -> dict:
    overlap_original = []
    overlap_generate = []

    no_workers = 100
    batch_size = len(items) // no_workers + 1
    batches = [items[i * batch_size: (i + 1) * batch_size] for i in range(no_workers)]

    with Pool(no_workers) as pool:
        results = pool.map_async(_worker, batches)
        pool.close()
        pool.join()

        for result in results.get():
            overlap_original.extend(result[0])
            overlap_generate.extend(result[1])

    return {
        "mse": str(mean_squared_error(overlap_original, overlap_generate)),
        "mae": str(mean_absolute_error(overlap_original, overlap_generate)),
        "spearman": spearmanr(overlap_generate, overlap_original),
        "pearson": pearsonr(overlap_generate, overlap_original)
    }


def eval_all_generate(path_director: str) -> None:
    results = dict()

    for path_director in tqdm(Path(path_director).glob("**/sample_output/")):
        results_director = dict()

        for path in tqdm(Path(path_director).glob("*.json")):

            if 'top-k' in path.name:
                continue

            with path.open('r') as file_input:
                data = json.load(file_input)

            if 'overlap' not in data[0]:
                continue

            results_director[path.name.replace('.json', '')] = eval_items(data)
        if not results_director:
            continue

        results[path_director.parent.name] = results_director

    with open('result_overlap.json', 'w+') as file_output:
        json.dump(results, file_output, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    eval_all_generate("/home/mihai/Desktop/backup/genaration/genaration")
