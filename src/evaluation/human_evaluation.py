import json

import pandas as pd
import pingouin as pg
from statistics import mean

if __name__ == '__main__':
    path_csv = "/home/mihai/Desktop/result-xl/file2.tsv"
    path_to_text = "/home/mihai/Desktop/RoGPT2-Application/test/selected.json"
    df = pd.read_csv(path_csv, sep='\t')
    list_summary_news: dict = json.load(open(path_to_text, 'r'))
    keys = list(list_summary_news.keys())
    user_name = {0: "userA", 1: "userB", 2: "userC", 3: "userD", 4: "userE", 5: "userF"}

    scores = dict()
    for index, row in df.iterrows():
        count = 0

        scores_users = dict()
        for type_model in keys:
            scores_metrics = {"Ideea": list(), "Detalii": list(), "Coeziune": list(), "Parafrazarea": list(),
                              "Limbaj": list()}
            for i in range(5):
                for type_metrics in ["Ideea", "Detalii", "Coeziune", "Parafrazarea", "Limbaj"]:
                    scores_metrics[type_metrics].append(int(row[count]))

                    count += 1

            scores_users[type_model] = scores_metrics
        scores[user_name[int(index)]] = scores_users
    print(scores)

    scores_type_models = dict()
    for type_model in keys:
        scores_type_models[type_model] = dict()

    for user, evaluation in scores.items():
        for type_model, eval_score in evaluation.items():
            scores_type_models[type_model].update({user: eval_score})
    print(scores_type_models)

    final_results = dict()
    scores_icc_all = dict()
    for type_model, evaluation in scores_type_models.items():
        scores_icc = {"user": list(), "type_metrics": list(), "score": list()}
        scores_metrics = {"Ideea": list(), "Detalii": list(), "Coeziune": list(), "Parafrazarea": list(),
                          "Limbaj": list()}
        for user, user_evaluation in evaluation.items():
            for metric, scores_metric in user_evaluation.items():
                scores_icc["user"].append(user)
                scores_icc["type_metrics"].append(metric)
                mean_score = mean(scores_metric)
                # print(mean_score, scores_metric, type_model, metric, user)
                scores_icc["score"].append(mean_score)
                scores_metrics[metric] += scores_metric

        scores_icc_all[type_model] = scores_icc

        final_results[type_model] = dict()
        for metrics, scores in scores_metrics.items():
            # print(metrics, scores, type_model)
            final_results[type_model][metrics] = mean(scores)

    print(scores_icc_all)
    for type_model, map_icc in scores_icc_all.items():
        icc = pg.intraclass_corr(data=pd.DataFrame.from_dict(map_icc), targets='type_metrics', raters='user',
                                 ratings='score')
        icc.set_index("Type")
        # print(icc)
        final_results[type_model]["icc3_1"] = icc["CI95%"][2][1]
        final_results[type_model]["icc3_k"] = icc["CI95%"][5][1]
        final_results[type_model]["icc2_1"] = icc["CI95%"][1][1]
        final_results[type_model]["icc2_k"] = icc["CI95%"][4][1]

    print(final_results)