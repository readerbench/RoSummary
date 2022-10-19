import json
from pathlib import Path
from rouge import Rouge
from statistics import mean

if __name__ == '__main__':
    rouge = Rouge()
    path_text_generate = "/home/mihai/Desktop/models/fep/RoGPT2-Application/genaration/summary/sample_output"
    result = dict()

    for path in Path(path_text_generate).glob("*.json"):
        with path.open('r') as file_input:
            data = json.load(file_input)

        print(path.name)
        rouge_1 = list()
        rouge_2 = list()
        rouge_l = list()

        for example in data:
            original = example['original']
            generate = example['generate']
            score = rouge.get_scores(original, generate, avg=True)
            rouge_1.append(score['rouge-1']['f'])
            rouge_2.append(score['rouge-2']['f'])
            rouge_l.append(score['rouge-l']['f'])

        result[path.name.replace(".json", "")] = {'rouge-1': mean(rouge_1), "rouge-2": mean(rouge_2),
                                                  "rouge-l": mean(rouge_l)}

    print(json.dumps(result, indent=4))

