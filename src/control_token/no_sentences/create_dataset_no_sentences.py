from __future__ import annotations

import random

import spacy
from transformers import GPT2Tokenizer

from src.utils.utils import create_dataset_control_token, create_dataset_control_token_test

model_ro = spacy.load('ro_core_news_lg')
tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')


def processing_aleph_no_sentences(item: dict) -> list[str]:
    sentences_summary = [sentence.text for paragraph in item['summary'] for sentence in model_ro(paragraph).sents]

    if len(sentences_summary) < 3:
        line = f'NoSentences: {len(sentences_summary)} Text: {item["title"]} {" ".join(item["paragraphs"])} ' \
               f'Summary: {" ".join(sentences_summary)}{tokenizer.eos_token}'
        return [line]

    index_start = random.choice(range(int(0.35 * len(sentences_summary)), int(0.8 * len(sentences_summary))))
    lines = list()

    for i in range(index_start, len(sentences_summary)):
        selected_summary = sentences_summary[:(i + 1)]
        line_text = f'NoSentences: {len(selected_summary)} Text: {item["title"]} {" ".join(item["paragraphs"])} ' \
                    f'Summary: {" ".join(selected_summary)}{tokenizer.eos_token}'
        lines.append(line_text)

    return lines


def processing_aleph_no_sentences_test(item: dict) -> list[dict]:
    sentences_summary = [sentence.text for paragraph in item['summary'] for sentence in model_ro(paragraph).sents]

    return [{
        "no-sentences": len(sentences_summary),
        'text': f'{item["title"]} {" ".join(item["paragraphs"])}',
        'summary': " ".join(sentences_summary)
    }]


if __name__ == '__main__':
    # create_dataset_control_token("../../../dataset/aleph_news/split_block", "../../../dataset/aleph_news/no_sentences",
    #                              processing_aleph_no_sentences, False)

    create_dataset_control_token_test("../../../dataset/aleph_news/split",
                                      "../../../dataset/aleph_news/no_sentences", processing_aleph_no_sentences_test,
                                      False)