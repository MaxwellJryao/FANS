from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from dataclasses import dataclass, field
from transformers import HfArgumentParser
import json
import utils

@dataclass
class Arguments:
    translate_model_name_or_path: str = field(default="AI-MO/Kimina-Autoformalizer-7B")
    tensor_parallel_size: int = field(default=1)
    gpu_memory_utilization: float = field(default=0.8)
    result_dir: str = field(default="results")
    dataset_name: str = field(default="amc23")
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Math-1.5B-Instruct")

    temperature: float = field(default=0.6)
    top_p: float = field(default=0.95)
    max_new_tokens: int = field(default=2048)

    num_workers: int = field(default=1)
    local_rank: int = field(default=0)

problem_ans_template = """{problem} The answer is {answer}."""

translate_template = "Please autoformalize the following problem in Lean 4 with a header. Use the following theorem names: my_favorite_theorem.\n\n"

def main(args: Arguments):
    model = LLM(args.translate_model_name_or_path, tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.gpu_memory_utilization)
    
    result_file = f"{args.result_dir}/{args.dataset_name}/{args.model_name_or_path.split('/')[-1]}.json"
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    local_start, local_end = utils.alloc_data(args.local_rank, args.num_workers, len(results))
    results = results[local_start:local_end]

    tokenizer = AutoTokenizer.from_pretrained(args.translate_model_name_or_path, trust_remote_code=True)

    prompts = []
    for result in results:
        problem = result["problem"]
        for pred in result["extracted_preds"]:
            prompt = problem_ans_template.format(problem=problem, answer=pred)
            prompt = translate_template + prompt
            messages = [
                {"role": "system", "content": "You are an expert in mathematics and Lean 4."},
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)

    sampling_params = SamplingParams(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_new_tokens)
    outputs = model.generate(prompts, sampling_params=sampling_params)

    for i, result in enumerate(results):
        result["fl"] = []
        for j in range(len(result["extracted_preds"])):
            result["fl"].append(outputs[i * len(result["extracted_preds"]) + j].outputs[0].text)

    with open(f"{args.result_dir}/{args.dataset_name}/{args.model_name_or_path.split('/')[-1]}_{args.local_rank}.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args()
    main(args)
