# RoSummary


This is a version of the RoGPT2 model trained on the [AlephNews](https://huggingface.co/datasets/readerbench/AlephNews) dataset for the summarization task. There are 3 trained versions, they are available on the HuggingFace Hub:
* [base](https://huggingface.co/readerbench/RoSummary-base)
* [medium](https://huggingface.co/readerbench/RoSummary-medium)
* [large](https://huggingface.co/readerbench/RoSummary-large)


## Evaluation on [AlephNews](https://huggingface.co/datasets/readerbench/AlephNews)


| Model  | Decode Method  |           | BERTScore |          |          |   ROUGE  |          |
|:------:|:--------------:|:---------:|:---------:|:--------:|:--------:|:--------:|:--------:|
| 	 |                | Precision |   Recall  | F1-Score |  ROUGE-1 |  ROUGE-2 |  ROUGE-L |
| 	 |	Greedy	  |   0.7335  |   0.7399  |  0.7358  |  0.3360  |  0.1862  |  0.3333  |
| Base   |  Beam Search	  |   0.7354  |   0.7468  |  0.7404  |  0.3480  |  0.1991  |  0.3416  |
|	 | Top-p Sampling |   0.7296  |   0.7299  |  0.7292  |  0.3058  |  0.1452  |  0.2951  |
|	 |	Greedy	  |   0.7378  |   0.7401  |  0.7380  |  0.3422  |  0.1922  |  0.3394  |
| Medium |  Beam Search	  |   0.7390  | **0.7493**|**0.7434**|**0.3546**|**0.2061**|**0.3467**|
|	 | Top-p Sampling |   0.7315  |   0.7285  |  0.7294  |  0.3042  |  0.1400  |  0.2921  |
|	 |	Greedy	  |   0.7376  |   0.7424  |  0.7391  |  0.3414  |  0.1895  |  0.3355  |
| Large	 |  Beam Search	  | **0.7394**|   0.7470  |  0.7424  |  0.3492  |  0.1995  |  0.3384  |
|	 | Top-p Sampling |   0.7311  |   0.7301  |  0.7299  |  0.3051  |  0.1418  |  0.2931  |

You can find a Jupyter Notebook where there is a [demo](DemoRoSummary.ipynb) for RoSummary and AlephNews.

































