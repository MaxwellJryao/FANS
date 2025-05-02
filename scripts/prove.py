from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from vllm import LLM, SamplingParams
import json
import re

@dataclass
class Arguments:
    prover_model_name_or_path: str = field(default="deepseek-ai/DeepSeek-Prover-V2-7B")
    tensor_parallel_size: int = field(default=1)
    gpu_memory_utilization: float = field(default=0.8)

    max_new_tokens: int = field(default=8192)

    result_dir: str = field(default="results")
    dataset_name: str = field(default="amc23")
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Math-1.5B-Instruct")

    seed: int = field(default=30)
    num_workers: int = field(default=1)
    local_rank: int = field(default=0)

formal_statement_template = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- {nl_problem} -/
{fl_problem}
""".strip()

prompt_template = """
Complete the following Lean 4 code:

```lean4
{formal_statement}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

def extract_proof(outputs):
    pattern = r"```lean4\s+(.*?)```"
    matches = re.findall(pattern, outputs, re.DOTALL)
    if matches:
        return matches[-1]
    else:
        return outputs

def main(args: Arguments):
    torch.manual_seed(args.seed)
    result_file = f"{args.result_dir}/{args.dataset_name}/{args.model_name_or_path.split('/')[-1]}.json"
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    local_start = args.local_rank * (len(results) // args.num_workers)
    if args.local_rank != args.num_workers - 1:
        local_end = local_start + (len(results) // args.num_workers)
    else:
        local_end = len(results)
    results = results[local_start:local_end]

    tokenizer = AutoTokenizer.from_pretrained(args.prover_model_name_or_path)

    model = LLM(args.prover_model_name_or_path, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)

    prompts = []    
    for result in results:
        nl_problem = result["problem"]
        for fl_problem in result["fl"]:
            formal_statement = formal_statement_template.format(nl_problem=nl_problem, fl_problem=fl_problem)
            chat = [
                {"role": "user", "content": prompt_template.format(formal_statement=formal_statement)},
            ]
            inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            prompts.append(inputs)

    sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_new_tokens)
    outputs = model.generate(prompts, sampling_params=sampling_params)

    for i, result in enumerate(results):
        result["proofs"] = []
        result['all_proofs'] = []
        for j in range(len(result["fl"])):
            result["proofs"].append(extract_proof(outputs[i * len(result["fl"]) + j].outputs[0].text))
            result['all_proofs'].append(outputs[i * len(result["fl"]) + j].outputs[0].text)

    with open(f'{args.result_dir}/{args.dataset_name}/{args.model_name_or_path.split("/")[-1]}_{args.local_rank}.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args()
    main(args)
