from vllm import LLM, SamplingParams
from dataclasses import dataclass, field
from transformers import HfArgumentParser
from datasets import load_dataset
import json
import utils

@dataclass
class Arguments:
    # model config
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    tensor_parallel_size: int = field(default=1)
    gpu_memory_utilization: float = field(default=0.8)

    # dataset config
    dataset_path: str = field(default="HuggingFaceH4/MATH-500")
    dataset_name: str = field(default="math500")
    dataset_split: str = field(default="test")
    dataset_end: int = field(default=1000)

    # generation config
    max_new_tokens: int = field(default=4096)
    temperature: float = field(default=0.7)
    top_p: float = field(default=0.95)
    n: int = field(default=8)

    num_workers: int = field(default=1)
    local_rank: int = field(default=0)

def main(args: Arguments):
    ds = load_dataset(args.dataset_path)[args.dataset_split]
    if args.dataset_end and args.dataset_end > 0:
        ds = ds.select(range(min(args.dataset_end, len(ds))))
    print(f"Loaded {len(ds)} examples from {args.dataset_path} {args.dataset_split}")

    local_start, local_end = utils.alloc_data(args.local_rank, args.num_workers, len(ds))
    ds = ds.select(range(local_start, local_end))
    print(f"Rank {args.local_rank} processing {len(ds)} examples from {local_start} to {local_end}")

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
    )

    prompts = []
    if args.dataset_name in ["math500", "minerva_math", "olympiad_bench", "amc23", "aime24"]:
        for item in ds:
            message = [
                {'role': 'system', 'content': 'Please reason step by step, and put your final answer within \\boxed{}.'},
                {'role': 'user', 'content': item['problem']}
            ]
            prompts.append(message)
    else:
        raise ValueError(f"Dataset {args.dataset_name} not supported")

    results = llm.chat(prompts, sampling_params)

    outputs = []
    if args.dataset_name == 'math500':
        for i in range(len(ds)):
            outputs.append({
                'problem': ds[i]['problem'],
                'preds': [result.text for result in results[i].outputs],
                'gt': ds[i]['answer'],
                'solution': ds[i]['solution'],
                'subject': ds[i]['subject'],
                'level': ds[i]['level']
            })
    elif args.dataset_name == 'minerva_math':
        for i in range(len(ds)):
            outputs.append({
                'problem': ds[i]['problem'],
                'preds': [result.text for result in results[i].outputs],
                'gt': ds[i]['answer'],
                'solution': ds[i]['answer'],
                'type': ds[i]['type']
            })
    elif args.dataset_name == 'olympiad_bench':
        for i in range(len(ds)):
            outputs.append({
                'problem': ds[i]['problem'],
                'preds': [result.text for result in results[i].outputs],
                'gt': ds[i]['answer'][0],
                'solution': ds[i]['answer'],
                'subfield': ds[i]['subfield']
            })
    elif args.dataset_name == 'amc23':
        for i in range(len(ds)):
            outputs.append({
                'problem': ds[i]['problem'],
                'preds': [result.text for result in results[i].outputs],
                'gt': ds[i]['answer']
            })
    elif args.dataset_name == 'aime24':
        for i in range(len(ds)):
            outputs.append({
                'problem': ds[i]['problem'],
                'preds': [result.text for result in results[i].outputs],
                'gt': ds[i]['answer'],
                'solution': ds[i]['answer']
            })
    
    save_path = f'results/{args.dataset_name}/{args.model_name_or_path.split("/")[-1]}_{args.local_rank}.json'
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

    print(f"Results saved to {save_path}")
    

if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args()
    main(args)