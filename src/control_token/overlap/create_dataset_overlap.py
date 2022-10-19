from __future__ import annotations

import spacy
import random

from nltk import ngrams

from src.utils.utils import create_dataset_control_token, create_dataset_control_token_test

model_ro = spacy.load('ro_core_news_lg')


def processing_aleph_news_overlapping(item: dict) -> list[str]:
    nlp_summary = [token.text for token in model_ro(" ".join(item['summary'])) if (not token.is_stop) and
                   (not token.is_punct)]
    nlp_news = [token.text for token in model_ro(item["title"] + " ".join(item['paragraphs'])) if (not token.is_stop)
                and (not token.is_punct)]

    ng_4_summary = set(list(ngrams(nlp_summary, 4)))
    ng_4_news = set(list(ngrams(nlp_news, 4)))
    count_overlap = 0

    for gram in ng_4_summary:
        if gram in ng_4_news:
            count_overlap += 1

    line = f'LexOverlap: {(count_overlap / len(nlp_summary)):.4f} Text: {item["title"]} {" ".join(item["paragraphs"])} ' \
           f'Summary: {" ".join(item["summary"])}<|endoftext|>'

    return [line]


def processing_aleph_news_overlapping_test(item: dict) -> list[dict]:
    nlp_summary = [token.text for token in model_ro(" ".join(item['summary'])) if (not token.is_stop) and
                   (not token.is_punct)]
    nlp_news = [token.text for token in model_ro(item["title"] + " ".join(item['paragraphs'])) if (not token.is_stop) and
                   (not token.is_punct)]

    ng_4_summary = set(list(ngrams(nlp_summary, 4)))
    ng_4_news = set(list(ngrams(nlp_news, 4)))
    count_overlap = 0

    for gram in ng_4_summary:
        if gram in ng_4_news:
            count_overlap += 1

    return [{
        "overlap": f'{(count_overlap / len(nlp_summary)):.4f}',
        "text": f'{item["title"]} {" ".join(item["paragraphs"])}',
        "summary": " ".join(item["summary"])
    }]


if __name__ == '__main__':
    # create_dataset_control_token("../../../dataset/aleph_news/split_block", "../../../dataset/aleph_news/overlap-fix",
    #                              processing_aleph_news_overlapping, False)

    create_dataset_control_token_test("../../../dataset/aleph_news/split",
                                      "../../../dataset/aleph_news/overlap-fix", processing_aleph_news_overlapping_test,
                                      False)

