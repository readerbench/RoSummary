from __future__ import division, unicode_literals

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


if __name__ == '__main__':
    sentence1 = 'este o zi de vara'
    sentence2 = 'este o zi ploiasa'
    compute_bert_score(sentence1, sentence2)
