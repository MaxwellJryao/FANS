from dataclasses import dataclass, field
import json
from transformers import HfArgumentParser
from tqdm import tqdm
from datasets import load_dataset

@dataclass
class Arguments:
    result_dir: str = field(default="results")
    dataset_name: str = field(default="aime24")
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    subfield: str = field(default="Number Theory")
    # olympiad_bench: Number Theory, Algebra, Combinatorics, Geometry
    # math500: Precalculus, Algebra, Geometry, Intermediate Algebra, Prealgebra, Counting & Probability, Number Theory

    mode: str = field(default="direct_majority_vote", metadata={"choices": ["majority_vote", "average", "direct_pass", "direct_majority_vote"]})

olympiad_bench_subfields = ["Number Theory", "Algebra", "Combinatorics", "Geometry"]
math500_subfields = ["Precalculus", "Algebra", "Geometry", "Intermediate Algebra", "Prealgebra", "Counting & Probability", "Number Theory"]

def main(args: Arguments):
    ds = load_dataset('FlippyDora/{}'.format(args.dataset_name))['train']
    result_file = f'{args.result_dir}/{args.dataset_name}/{args.model_name_or_path.split("/")[-1]}.json'

    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    correct = 0
    total = 0
    use_fl = 0
    for problem_id, result in enumerate(results):
        # if args.dataset_name == "math500" and ds[problem_id]['subject'] != args.subfield:
        #     continue
        # elif args.dataset_name == "olympiad_bench" and ds[problem_id]['subfield'] not in args.subfield:
        #     continue
        total += 1
        cur_correct = 0
        found = False
        if 'direct' not in args.mode:
            cur_list = {}
            for i, fl_result in enumerate(result['fl_verify_results']):
                if (not fl_result['has_error']) and fl_result['is_valid_no_sorry'] and (result['proofs'][i] != "") \
                    and ('theorem' in result['proofs'][i]) and (not isinstance(result['extracted_preds'][i], list)) \
                    and (result['fl_check_res'][i]) \
                    and (len(result['fl'][i].split('=')) > 1) and (result['extracted_preds'][i] in result['fl'][i].split('=')[-2]):
                    if result['scores'][i] == 0:
                        continue
                    if result['extracted_preds'][i] not in cur_list:
                        cur_list[result['extracted_preds'][i]] = 0
                    cur_list[result['extracted_preds'][i]] += 1
                    found = True
            if found:
                use_fl += 1
                sorted_list = sorted(cur_list.items(), key=lambda x: x[1], reverse=True)
                for i, pred in enumerate(result['extracted_preds']):
                    if pred == sorted_list[0][0] and result['scores'][i] == 1:
                        cur_correct = 1
                        break

        if not found:
            # fallback to majority vote, etc.
            if 'majority_vote' in args.mode:
                cur_list = {}
                for i, pred in enumerate(result['extracted_preds']):
                    if isinstance(pred, list):
                        pred = ''
                    if pred not in cur_list:
                        cur_list[pred] = 0
                    cur_list[pred] += 1

                sorted_list = sorted(cur_list.items(), key=lambda x: x[1], reverse=True)
                for i, pred in enumerate(result['extracted_preds']):
                    if pred == sorted_list[0][0] and result['scores'][i] == 1:
                        cur_correct = 1
                        break
            elif 'pass' in args.mode:
                cur_correct = (sum(result['scores']) > 0)
            

        correct += cur_correct

    # print(f"Dataset: {args.dataset_name}, Subfield: {args.subfield}, Mode: {args.mode}, Accuracy: {correct / total}, Use FL: {use_fl} {use_fl / total}")
    print(f"Dataset: {args.dataset_name}, Mode: {args.mode}, Accuracy: {correct / total}, Use FL: {use_fl} {use_fl / total}")


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args()
    for dataset_name in ["math500"]:
        # for subfield in math500_subfields if dataset_name == "math500" else olympiad_bench_subfields:
        for mode in ['direct_majority_vote', 'majority_vote']:
            args.dataset_name = dataset_name
            args.mode = mode
            # args.subfield = subfield
            main(args)
