from __future__ import annotations

import spacy
import random

from nltk import ngrams
from transformers import GPT2Tokenizer

from src.utils.utils import create_dataset_control_token_test, create_dataset_control_token

model_ro = spacy.load('ro_core_news_lg')
tokenizer = GPT2Tokenizer.from_pretrained('readerbench/RoGPT2-large')


def processing_aleph_overlap_no_words(item: dict) -> list[str]:
    sentences_summary = [sentence.text for paragraph in item['summary'] for sentence in model_ro(paragraph).sents]
    nlp_news = [token.text for token in model_ro(item["title"] + " ".join(item['paragraphs'])) if not token.is_stop or
                not token.is_punct]

    if len(sentences_summary) < 3:
        words = [token.text for token in model_ro(" ".join(sentences_summary)) if not token.is_punct]
        nlp_summary = [token.text for token in model_ro(" ".join(item['summary'])) if not token.is_stop or
                       not token.is_punct]
        ng_4_summary = set(list(ngrams(nlp_summary, 4)))
        ng_4_news = set(list(ngrams(nlp_news, 4)))
        count_overlap = 0

        for gram in ng_4_summary:
            if gram in ng_4_news:
                count_overlap += 1

        lines = [
            f'NoWords: {len(words)} Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: {" ".join(sentences_summary)}{tokenizer.eos_token}',
            f'LexOverlap: {(count_overlap / len(nlp_summary)):.4f} Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: {" ".join(sentences_summary)}{tokenizer.eos_token}',
            f'NoWords: {len(words)} LexOverlap: {(count_overlap / len(nlp_summary)):.4f} Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: {" ".join(sentences_summary)}{tokenizer.eos_token}',
            f'LexOverlap: {(count_overlap / len(nlp_summary)):.4f} NoWords: {len(words)} Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: {" ".join(sentences_summary)}{tokenizer.eos_token}',
        ]
        # LexOverlap:

        return lines

    index_start = random.choice(range(int(0.35 * len(sentences_summary)), int(0.8 * len(sentences_summary))))
    lines = list()

    for i in range(index_start, len(sentences_summary)):
        selected_summary = sentences_summary[:(i + 1)]
        words = [token.text for token in model_ro(" ".join(selected_summary)) if not token.is_punct]

        nlp_summary = [token.text for token in model_ro(" ".join(item['summary'])) if not token.is_stop or
                       not token.is_punct]
        ng_4_summary = set(list(ngrams(nlp_summary, 4)))
        ng_4_news = set(list(ngrams(nlp_news, 4)))
        count_overlap = 0

        for gram in ng_4_summary:
            if gram in ng_4_news:
                count_overlap += 1

        lines_text = [
            f'NoWords: {len(words)} Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: {" ".join(sentences_summary)}{tokenizer.eos_token}',
            f'LexOverlap: {(count_overlap / len(nlp_summary)):.4f} Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: {" ".join(sentences_summary)}{tokenizer.eos_token}',
            f'NoWords: {len(words)} LexOverlap: {(count_overlap / len(nlp_summary)):.4f} Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: {" ".join(sentences_summary)}{tokenizer.eos_token}',
            f'LexOverlap: {(count_overlap / len(nlp_summary)):.4f} NoWords: {len(words)} Text: {item["title"]} {" ".join(item["paragraphs"])} Summary: {" ".join(sentences_summary)}{tokenizer.eos_token}',
        ]

        lines.extend(lines_text)

    return lines


def processing_aleph_test(item: dict) -> list[dict]:
    sentences_summary = [sentence.text for paragraph in item['summary'] for sentence in model_ro(paragraph).sents]
    words = [token.text for token in model_ro(" ".join(sentences_summary)) if not token.is_punct]

    nlp_summary = [token.text for token in model_ro(" ".join(item['summary'])) if not token.is_stop or
                   not token.is_punct]
    nlp_news = [token.text for token in model_ro(item["title"] + " ".join(item['paragraphs'])) if not token.is_stop or
                not token.is_punct]

    ng_4_summary = set(list(ngrams(nlp_summary, 4)))
    ng_4_news = set(list(ngrams(nlp_news, 4)))
    count_overlap = 0

    for gram in ng_4_summary:
        if gram in ng_4_news:
            count_overlap += 1

    token_news = tokenizer.encode(item["title"] + " ".join(item["paragraphs"]))
    token_summary = tokenizer.encode(" ".join((item["summary"])))

    return [{
        "title": item['title'],
        'paragraphs': item['paragraphs'],
        'summary-list': item['summary'],
        "text": f"{item['title']} {' '.join(item['paragraphs'])}",
        "summary": " ".join(sentences_summary),
        "no-words": len(words),
        # "no-sentences": len(sentences_summary),
        "overlap": f'{(count_overlap / len(nlp_summary)):.4f}',
        # "ration-tokens": f'{(len(token_summary) / len(token_news)):.4f}'
    }]


if __name__ == '__main__':
    # create_dataset_control_token_test("../../../dataset/aleph_news/split",
    #                                   "../../../dataset/aleph_news/complex-all4", processing_aleph_test,
    #                                   False)

    create_dataset_control_token_test("../../../dataset/aleph_news/split",
                                      "../../../dataset/aleph_news/combine-no-words-overlap", processing_aleph_test,
                                      False)

    # create_dataset_control_token("../../../dataset/aleph_news/split", "../../../dataset/aleph_news/all-no-words-overlap",
    #                              processing_aleph_overlap_no_words, False)
