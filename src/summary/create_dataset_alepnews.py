from __future__ import annotations

import os
import json
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from multiprocessing import Pool
from random import shuffle, randrange

import spacy
from transformers import GPT2Tokenizer

from src.utils.utils import clean_text

model_ro = spacy.load('ro_core_news_lg')
tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')


def split_dataset(path_raw_dataset: str, path_dataset_split: str) -> None:
    items = list()
    items_under = list()

    for path in tqdm(Path(path_raw_dataset).glob("**/*.json")):
        with path.open('r') as file_input:
            data = json.load(file_input)

        for elem in data:
            summary = list()
            paragraphs = list()

            if len(elem['summary']) == 0:
                continue

            if len(elem['paragraphs']) == 0 or len(" ".join(elem['paragraphs'])) <= 20:
                continue

            elem["title"] = clean_text(elem["title"])

            for line_summary in elem['summary']:
                line_summary_clean = clean_text(line_summary)

                if len(line_summary_clean) == 0:
                    continue
                summary.append(line_summary_clean)

            for line_paragraph in elem['paragraphs']:
                line_paragraph_clean = clean_text(line_paragraph)

                if len(line_paragraph_clean) == 0:
                    continue
                paragraphs.append(line_paragraph_clean)

            if len(summary) == 0 or len(paragraphs) == 0:
                continue

            if len(tokenizer.encode(elem["title"] + " " + " ".join(paragraphs))) <= 715:
                items_under.append({
                    "url": elem['url'],
                    "title": elem['title'],
                    "summary": summary,
                    "paragraphs": paragraphs
                })
            else:
                items.append({
                    "url": elem['url'],
                    "title": elem['title'],
                    "summary": summary,
                    "paragraphs": paragraphs
                })

    len_total = len(items) + len(items_under)
    test_len = int(len_total * 0.05)
    shuffle(items_under)
    test = [items_under.pop(randrange(len(items_under))) for _ in range(test_len)]
    items += items_under
    train_index = int(0.95 * len(items))
    shuffle(items_under)
    train, dev = items[:train_index], items[train_index:]

    print(train_index, test_len)

    if not os.path.exists(path_dataset_split):
        os.makedirs(path_dataset_split)

    with open(os.path.join(path_dataset_split, 'train.json'), 'w+') as file_output:
        json.dump(train, file_output, indent=4, ensure_ascii=False)

    with open(os.path.join(path_dataset_split, 'dev.json'), 'w+') as file_output:
        json.dump(dev, file_output, indent=4, ensure_ascii=False)

    with open(os.path.join(path_dataset_split, 'test.json'), 'w+') as file_output:
        json.dump(test, file_output, indent=4, ensure_ascii=False)


def create_dataset_split_news(path_dataset: str, path_dataset_split_news: str, context_size: int = 724,
                              max_create_news: int = 3) -> None:
    if not os.path.exists(path_dataset_split_news):
        os.makedirs(path_dataset_split_news)

    for path in Path(path_dataset).glob("**/*.json"):
        result = list()

        if 'test.json' in str(path):
            continue

        with path.open('r') as file_input:
            data = json.load(file_input)

        debug_count = 0
        for item in tqdm(data):
            text = f'Text: {item["title"]} {" ".join(item["paragraphs"])}'
            token_text = tokenizer.encode(text)
            summary = f' Summary: {" ".join(item["summary"])}{tokenizer.eos_token}'
            tokens_summary = tokenizer.encode(summary)

            if len(token_text) + len(tokens_summary) >= context_size + 1:
                count = 0
                debug_count += 1
                sentences = [sentence.text for paragraph in item['paragraphs']
                             for sentence in model_ro(paragraph).sents]
                text_paragraphs = f"Text: {item['title']}"
                tokens_text = tokenizer.encode(text_paragraphs)

                current_news = []
                while sentences:
                    current_paragraph = sentences.pop(0)
                    current_tokens = tokenizer.encode(current_paragraph)

                    next_paragraph = sentences[0] if len(sentences) >= 1 else ""
                    next_tokens = tokenizer.encode(next_paragraph) if len(sentences) >= 1 else []

                    if len(current_paragraph) + len(tokens_text) + len(tokens_summary) + \
                            len(next_tokens) > context_size + 1:
                        result.append({
                            "url": item['url'],
                            "title": item['title'],
                            "summary": item['summary'],
                            "paragraphs": current_news + [current_paragraph]
                        })
                        count += 1

                        if count >= max_create_news:
                            break
                    else:
                        text_paragraphs = f"{text_paragraphs} {current_paragraph}"
                        tokens_text += current_tokens
                        current_news.append(current_paragraph)

            else:
                result.append(item)

        print(f'For partition: {path.name}, size: {len(result)}, count new: {debug_count}')

        with open(os.path.join(path_dataset_split_news, path.name), "w+") as file_output:
            json.dump(result, file_output, ensure_ascii=False, indent=4)


def write_dataset_summary(path_dataset: str, path_dataset_output: str) -> None:
    if not os.path.exists(path_dataset_output):
        os.makedirs(path_dataset_output)

    for path in tqdm(Path(path_dataset).glob("**/*.json")):
        with path.open('r') as file_input:
            data = json.load(file_input)

        with open(os.path.join(path_dataset_output, path.name.replace('.json', '.txt')), "w+") as file_output:
            for item in data:
                line = f'Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: {" ".join(item["summary"])}{tokenizer.eos_token}'
                file_output.write(f'{line}\n')


if __name__ == '__main__':
    # split_dataset("../../dataset/aleph_news/data", "../../dataset/aleph_news/split")
    # create_dataset_split_news("../../dataset/aleph_news/split", "../../dataset/aleph_news/split_block")
    write_dataset_summary("../../dataset/aleph_news/split_block", "../../dataset/aleph_news/summary")
