from __future__ import annotations

import random

import spacy
from transformers import GPT2Tokenizer

from src.utils.utils import create_dataset_control_token, create_dataset_control_token_test

model_ro = spacy.load('ro_core_news_lg')
tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')


def processing_aleph_no_words(item: dict) -> list[str]:
    sentences_summary = [sentence.text for paragraph in item['summary'] for sentence in model_ro(paragraph).sents]

    if len(sentences_summary) < 3:
        words = [token.text for token in model_ro(" ".join(sentences_summary)) if not token.is_punct]
        line = f'NoWords: {len(words)} Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: ' \
               f'{" ".join(sentences_summary)}{tokenizer.eos_token}'
        return [line]

    index_start = random.choice(range(int(0.35 * len(sentences_summary)), int(0.8 * len(sentences_summary))))
    lines = list()

    for i in range(index_start, len(sentences_summary)):
        selected_summary = sentences_summary[:(i + 1)]
        words = [token.text for token in model_ro(" ".join(selected_summary)) if not token.is_punct]
        line_text = f'NoWords: {len(words)} Text: {item["title"]} {" ".join(item["paragraphs"])} ' \
                    f'Summary: {" ".join(selected_summary)}{tokenizer.eos_token}'
        lines.append(line_text)

    return lines


def processing_aleph_no_words_test(item: dict) -> list[dict]:
    sentences_summary = [sentence.text for paragraph in item['summary'] for sentence in model_ro(paragraph).sents]
    words = [token.text for token in model_ro(" ".join(sentences_summary)) if not token.is_punct]

    return [{
        "text": f"{item['title']} {' '.join(item['paragraphs'])}",
        "summary": " ".join(sentences_summary),
        "no-words": len(words)
    }]


if __name__ == '__main__':
    # create_dataset_control_token("../../../dataset/aleph_news/split_block", "../../../dataset/aleph_news/no_words",
    #                              processing_aleph_no_words, False)

    create_dataset_control_token_test("../../../dataset/aleph_news/split",
                                      "../../../dataset/aleph_news/no_words", processing_aleph_no_words_test,
                                      False)