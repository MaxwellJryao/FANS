from dataclasses import dataclass, field
from transformers import HfArgumentParser
import json
import os

@dataclass
class Arguments:
    result_dir: str = field(default="results")
    dataset_name: str = field(default="math500")
    model_name_or_path: str = field(default="Qwen2.5-Math-1.5B-Instruct")
    num_files: int = field(default=1)

def main(args: Arguments):
    results = []
    prefix = f"{args.model_name_or_path.split('/')[-1]}"
    for i in range(args.num_files):
        with open(f"{args.result_dir}/{args.dataset_name}/{prefix}_{i}.json", 'r', encoding='utf-8') as f:
            results.extend(json.load(f))

    with open(f"{args.result_dir}/{args.dataset_name}/{prefix}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    # delete intermediate files
    for i in range(args.num_files):
        os.remove(f"{args.result_dir}/{args.dataset_name}/{prefix}_{i}.json")

if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args()
    main(args)