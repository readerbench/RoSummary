from __future__ import annotations

import re
import os
import json
import random
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor, wait

from transformers import GPT2Tokenizer

from typing import Callable


def clean_text(text_input: str) -> str:
    text = text_input.strip()

    text = text.replace(u'\xa0', u' ')
    text = text.replace('\u00ad', u' ')
    text = text.replace('\xad', '')
    text = text.replace('\u00ad', '')
    text = text.replace('\N{SOFT HYPHEN}', '')
    text = re.sub('==.*==', '', text)
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub('º|þ|È|™|Ó|Ñ|Ä|È|Ã|®|ƒ', '', text)
    text = re.sub('==*.', '', text)
    text = re.sub('♦', '', text)

    text = re.sub(" ,", ",", text)
    text = re.sub(" \.", ".", text)
    text = re.sub("[„”]", '"', text)
    text = re.sub("\[.*\]", '', text)
    text = re.sub("\n+", '\n', text)
    text = re.sub('[ ]+', ' ', text)
    text = re.sub('\t+', '\t', text)
    text = re.sub('\t', '', text)

    for remove_text in [
        'UPDATE.', 'UPDATE', 'GRAFICĂ. ', 'GRAFICĂ.', 'VIDEO.', 'VIDEO |', 'VIDEO', 'REPORTAJ', 'REPORTAJ.', 'AUDIO.',
        'AUDIO', 'Titlu:', 'TITLU:', 'Titlu', 'TITLU', 'FOTO.', 'FOTO', 'VIRAL. '
    ]:
        text = text.replace(remove_text, '')

    if len(text) != 0 and text[-1] not in ['.', '?', '!', ';', ':']:
        text = text + '.'

    return text


def _worker_create_dataset_control_token(args) -> list[str]:
    items, processing_item = args
    results = list()

    for item in tqdm(items):
        try:
            results.extend(processing_item(item))
        except Exception as e:
            print(f'Exception for pid:{os.getpid()} and item: {item}, Exception: {e}')
            return []

    return results


def create_dataset_control_token(path_dataset: str, path_dataset_output: str, processing_item: Callable,
                                 is_digi24: bool) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')
    print("Init tokenizer")
    if not os.path.exists(path_dataset_output):
        os.makedirs(path_dataset_output)

    for path in Path(path_dataset).glob("**/*.json"):
        if 'test' in path.name:
            continue

        with path.open('r') as file_input:
            data = json.load(file_input)

        if is_digi24:
            no_max_selected = 40_000 if 'train' in str(path.name) else 4_000  # for overlapping is 40000 with 4000
            selected = list()

            for item in tqdm(data):
                text = " ".join([item['title']] + item['text'])
                text_tokens = tokenizer.encode(text)

                if 700 >= len(text_tokens) > 300 and len(item['text']) >= 2 and len(item['text'][-1]) > 60:
                    selected.append(item)

            data = random.sample(selected, no_max_selected) if len(selected) > no_max_selected else selected

        no_workers = 20
        with open(os.path.join(path_dataset_output, path.name.replace('.json', '.txt')), "w+") as file_output, \
                Pool(no_workers) as pool:
            batch_size = len(data) // no_workers + 1
            batches = [(data[i * batch_size: (i + 1) * batch_size], processing_item) for i in range(no_workers)]

            results = pool.map_async(_worker_create_dataset_control_token, batches)
            pool.close()
            pool.join()

            for result in results.get():
                for line in result:
                    file_output.write(f'{line}\n')


def create_dataset_control_token_test(path_dataset: str, path_dataset_output: str, processing_item: Callable,
                                      is_digi24: bool) -> None:

    # tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-base')
    if not os.path.exists(path_dataset_output):
        os.makedirs(path_dataset_output)

    for path in Path(path_dataset).glob("**/*.json"):
        if 'test' not in path.name:
            continue

        with path.open('r') as file_input:
            data = json.load(file_input)

        results = list()

        # if is_digi24:
        #     no_max_selected = 3_000
        #     selected = list()
        #
        #     for item in tqdm(data):
        #         text = " ".join([item['title']] + item['text'])
        #         text_tokens = tokenizer.encode(text)
        #
        #         if 700 >= len(text_tokens) > 300 and len(item['text']) >= 2 and len(item['text'][-1]) > 60:
        #             selected.append(item)
        #
        #     data = random.sample(selected, no_max_selected) if len(selected) > no_max_selected else selected

        for item in tqdm(data):
            results.extend(processing_item(item))

        with open(os.path.join(path_dataset_output, path.name), "w+") as file_output:
            json.dump(results, file_output, ensure_ascii=False, indent=4)


def create_dataset_control_token_serial(path_dataset: str, path_dataset_output: str, processing_item: Callable,
                                        is_digi24: bool) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')
    print("Init tokenizer")
    if not os.path.exists(path_dataset_output):
        os.makedirs(path_dataset_output)

    for path in Path(path_dataset).glob("**/*.json"):
        if 'test' in path.name:
            continue

        with path.open('r') as file_input:
            data = json.load(file_input)

        if is_digi24:
            no_max_selected = 40_000 if 'train' in str(path.name) else 4_000  # for overlapping is 40000 with 4000
            selected = list()

            for item in tqdm(data):
                text = " ".join([item['title']] + item['text'])
                text_tokens = tokenizer.encode(text)

                if 700 >= len(text_tokens) > 300 and len(item['text']) >= 2 and len(item['text'][-1]) > 60:
                    selected.append(item)

            data = random.sample(selected, no_max_selected) if len(selected) > no_max_selected else selected

        with open(os.path.join(path_dataset_output, path.name.replace('.json', '.txt')), "w+") as file_output:
            for item in tqdm(data):
                for line in processing_item(item):
                    file_output.write(f'{line}\n')
