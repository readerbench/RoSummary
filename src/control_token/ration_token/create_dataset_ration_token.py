from __future__ import annotations

import random

import spacy
from transformers import GPT2Tokenizer

from src.utils.utils import create_dataset_control_token, create_dataset_control_token_test

model_ro = spacy.load('ro_core_news_lg')
tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')


def processing_aleph_ration_tokens(item: dict) -> list[str]:
    sentences_summary = [sentence.text for paragraph in item['summary'] for sentence in model_ro(paragraph).sents]

    if len(sentences_summary) < 3:
        token_summary = tokenizer.encode(" ".join(item['summary']))
        token_news = tokenizer.encode(item["title"] + " ".join(item["paragraphs"]))

        line = f'RatioTokens: {(len(token_summary) / len(token_news)):.4f} Text: {item["title"]} ' \
               f'{" ".join(item["paragraphs"])} Summary: {" ".join(item["summary"])}{tokenizer.eos_token}'

        return [line]

    index_start = random.choice(range(int(0.35 * len(sentences_summary)), int(0.8 * len(sentences_summary))))
    lines = list()

    token_news = tokenizer.encode(item["title"] + " ".join(item["paragraphs"]))
    for i in range(index_start, len(sentences_summary)):
        selected_summary = sentences_summary[:(i + 1)]
        token_summary = tokenizer.encode(" ".join(selected_summary))

        line_text = f'RatioTokens: {(len(token_summary) / len(token_news)):.4f} Text: {item["title"]} ' \
                    f'{" ".join(item["paragraphs"])} Summary: {" ".join(selected_summary)}{tokenizer.eos_token}'

        lines.append(line_text)

    return lines


def processing_aleph_ration_tokens_test(item: dict) -> list[dict]:
    token_summary = tokenizer.encode(" ".join(item['summary']))
    token_news = tokenizer.encode(item["title"] + " ".join(item["paragraphs"]))

    return [{
        "ration-tokens": f'{(len(token_summary) / len(token_news)):.4f}',
        "text": f'{item["title"]} {" ".join(item["paragraphs"])}',
        "Summary": " ".join(item["summary"])
    }]


if __name__ == '__main__':
    # create_dataset_control_token("../../../dataset/aleph_news/split_block", "../../../dataset/aleph_news/ration_tokens",
    #                              processing_aleph_ration_tokens, False)

    create_dataset_control_token_test("../../../dataset/aleph_news/split",
                                      "../../../dataset/aleph_news/ration_tokens", processing_aleph_ration_tokens_test,
                                      False)
