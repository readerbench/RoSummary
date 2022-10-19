from __future__ import annotations
import os
import json
import random

from tqdm import tqdm
from pathlib import Path
import argparse

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

from typing import List, Dict

PATH_DATASET_TEST = 'test.json'


def missing_generation(configuration_name: str, config_gen: dict, path_model: str) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')
    model = TFGPT2LMHeadModel.from_pretrained(path_model)
    max_generation = 500
    selected = []
    generation = []

    with open(PATH_DATASET_TEST, 'r') as file_input:
        data = json.load(file_input)

    for item in data:
        if len(item["text"]) == 0:
            continue

        sentence_text = f'Text: {item["title"]} {" ".join(item["text"])}'
        if len(tokenizer.encode(sentence_text)) > 820:
            continue

        selected.append(item)

    if len(selected) > max_generation:
        selected = random.sample(selected, max_generation)

    if not os.path.exists('sample_output'):
        os.mkdir('sample_output')

    for item in tqdm(selected):
        text = [item['title']] + item['text']

        sentence_input = f'Text: {" ".join(text)} Summary:'
        sentence_token = tokenizer.encode(sentence_input, return_tensors='tf')
        len_input_tokens = len(sentence_token[0])

        predict_tokens = model.generate(sentence_token, max_length=1024, pad_token_id=tokenizer.eos_token_id,
                                        **config_gen)[0][len_input_tokens:]
        predict_text = tokenizer.decode(predict_tokens).replace('<|endoftext|>', '')

        generation.append({
            "context":  text,
            "generate": predict_text
        })

    with open(os.path.join("sample_output", f'{configuration_name}-{path_model.split("/")[-1]}.json'), 'w+') as file_output:
        json.dump(generation, file_output, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()

    configs_gen = {
        'greedy': {},
        'beam-search-4': {'num_beams': 4, 'early_stopping': True},
        'sample-top-p': {'top_k': 25, 'top_p': 0.94, 'do_sample': True},
    }
    path_models = {
        'base': f'../../models/summary/model/RoGPT2-base-best',
        'medium': f'../../models/summary/model/RoGPT2-medium-best',
        'large': f'../../models/summary/model/RoGPT2-large-best'
    }

    missing_generation(args.config, configs_gen[args.config] , path_models[args.model])
