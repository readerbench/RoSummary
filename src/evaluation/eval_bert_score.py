from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from multiprocessing import Pool
from typing import Tuple, List, Any

from tqdm import tqdm
# from torchmetrics.text.bert import BERTScore

# bertscore = BERTScore(model_name_or_path="readerbench/RoBERT-large")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import TFAutoModel, BertTokenizer
from typing import Dict

tokenizer = BertTokenizer.from_pretrained('readerbench/RoBERT-large')
bert = TFAutoModel.from_pretrained('readerbench/RoBERT-large')


def compute_bert_score(generate: str, original: str) -> Dict[str, float]:
    tokens_generate = tokenizer.encode(generate, return_tensors='tf')
    tokens_original = tokenizer.encode(original, return_tensors='tf')
    embeddings_generate = bert(tokens_generate, return_dict=True).last_hidden_state[0]
    embeddings_original = bert(tokens_original, return_dict=True).last_hidden_state[0]

    cosine_sims = cosine_similarity(embeddings_original, embeddings_generate)
    max_columns = np.amax(cosine_sims, axis=0)  # for recall
    max_rows = np.amax(cosine_sims, axis=1)  # for precision

    precision = np.mean(max_rows)
    recall = np.mean(max_columns)
    f1 = 2 * (precision * recall) / (precision + recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def eval_bert_score_items(items: dict) -> dict:
    precision = list()
    recall = list()
    f1 = list()
    items = items[:50]

    for item in tqdm(items):
        if int(len(item["context"]) * 0.9) < len(item['original']):
            continue

        reference = item['original']
        score = compute_bert_score(item['generate'], reference) #bertscore([item['generate']], [reference])
        f1.append(score['f1'])
        precision.append(score['precision'])
        recall.append(score['recall'])

    return {
        "precision": str(mean(precision)),
        "recall": str(mean(recall)),
        "f1": str(mean(f1))
    }


def eval_all_generate(path_director: str) -> None:
    results = dict()
    count = 0
    for path_director in tqdm(Path(path_director).glob("**/sample_output/")):
        results_director = dict()

        for path in tqdm(Path(path_director).glob("*.json")):

            if 'top-k' in path.name:
                continue

            with path.open('r') as file_input:
                data = json.load(file_input)

            if 'original' not in data[0]:
                continue

            results_director[path.name.replace('.json', '')] = eval_bert_score_items(data)
            break

        if not results_director:
            continue

        results[path_director.parent.name] = results_director
        count += 1
        if count > 4:
            break

    with open('result_bertscore_new.json', 'w+') as file_output:
        json.dump(results, file_output, ensure_ascii=False, indent=4)


if __name__ == '__main__':
    eval_all_generate("../../genaration")