from math_verify.metric import math_metric
from math_verify.parser import LatexExtractionConfig, ExprExtractionConfig
import json
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from tqdm import tqdm

@dataclass
class Arguments:
    result_dir: str = field(default="results")
    dataset_name: str = field(default="math500")
    model_name_or_path: str = field(default="Qwen2.5-Math-7B-Instruct")

def compute_score(model_output: str, ground_truth: str, return_preds=False) -> bool:
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.

    str_preds = None
    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, str_preds = verify_func([ground_truth_boxed], [model_output])
    except:
        pass

    if return_preds:
        return ret_score, str_preds
    return ret_score

def main(args: Arguments):
    result_path = f"{args.result_dir}/{args.dataset_name}/{args.model_name_or_path.split('/')[-1]}.json"
    with open(result_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    for result in tqdm(results, desc="Scoring results"):
        if 'scores' in result:
            continue

        scores = []
        extracted_preds = []
        for i in range(len(result['preds'])):
            try:
                if isinstance(result['gt'], list):
                    result['gt'] = result['gt'][0]
                result['gt'] = str(result['gt'])
                score, extracted_pred = compute_score(result['preds'][i], result['gt'], return_preds=True)
                if extracted_pred and extracted_pred[1]:
                    extracted_pred = extracted_pred[1][-1]
                else:
                    extracted_pred = result['preds'][i]
            except:
                score = 0.
                extracted_pred = result['preds'][i]
            scores.append(score)
            extracted_preds.append(extracted_pred)
        result['scores'] = scores
        result['extracted_preds'] = extracted_preds

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args()
    main(args)