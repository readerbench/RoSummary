from __future__ import annotations
import os
import json
import random

from tqdm import tqdm
from pathlib import Path
import argparse

from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

from typing import List, Dict


def summary_no_sentences(configuration_name: str, config_gen: dict, path_model: str, dataset_path_test: str) -> None:
    tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')
    model = TFGPT2LMHeadModel.from_pretrained(path_model)

    generation = []
    
    with open(dataset_path_test, 'r') as file_input:
        data = json.load(file_input)

    if not os.path.exists('sample_output'):
        os.makedirs('sample_output', exist_ok=True)

    for item in tqdm(data):
        sentence_input = f'NoWords: {item["no-words"]} LexOverlap: {item["overlap"]} Text: {item["text"]} Summary:'
        sentence_token = tokenizer.encode(sentence_input, return_tensors='tf')
        len_input_tokens = len(sentence_token[0])

        predict_tokens = model.generate(sentence_token, max_length=1024, pad_token_id=tokenizer.eos_token_id,
                                        **config_gen)[0][len_input_tokens:]
        predict_text = tokenizer.decode(predict_tokens).replace('<|endoftext|>', '')

        generation.append({
            "context": item["text"],
            "original": item['summary'],
            "original-list": item['summary-list'],
            "generate": predict_text,
            "no-words": item["no-words"],
            "overlap" : item["overlap"]
        })

    with open(os.path.join("sample_output", f'{configuration_name}-{path_model.split("/")[-1]}.json'), 'w+') as file_output:
        json.dump(generation, file_output, indent=4, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    print(args.config)

    configs_gen = {
        'greedy': {},
        'beam-search-4': {'num_beams': 4, 'early_stopping': True},
        'sample-top-p': {'top_k': 25, 'top_p': 0.94, 'do_sample': True},
    }
    path_models = {
        'base': f'../../models/no-words-overlap-fix/model/RoGPT2-base-best',
        'medium': f'../../models/no-words-overlap-fix/model/RoGPT2-medium-best',
        'large': f'../../models/no-words-overlap-fix/model/RoGPT2-large-best'
    }

    summary_no_sentences(args.config, configs_gen[args.config] , path_models[args.model], 'test.json')
