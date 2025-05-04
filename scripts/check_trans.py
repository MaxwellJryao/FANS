from vllm import LLM, SamplingParams
from transformers import HfArgumentParser
from dataclasses import dataclass, field
import json
import utils

@dataclass
class Arguments:
    judge_model_name_or_path: str = field(default="Qwen/QwQ-32B")
    tensor_parallel_size: int = field(default=2)

    temperature: float = field(default=0.6)
    max_tokens: int = field(default=8192)

    result_dir: str = field(default="results")
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Math-1.5B-Instruct")
    dataset_name: str = field(default="math500")

    num_workers: int = field(default=1)
    local_rank: int = field(default=0)


template = """
**Task**:  
Determine whether the translated Lean4 theorem *exactly* matches the semantics of the original natural language math problem. Evaluate rigorously using the criteria below:  

**Evaluation Criteria**:  
1. **Variable Mapping**:  
   - Do numeric values/variables in the problem map correctly to Lean4 parameters?  
   - Example: "10 choose 4" must translate to `Nat.choose 10 4`, not `Nat.choose 4 10`.  
2. **Operator Correctness**:  
   - Are mathematical operations (e.g., combinations, equations, inequalities) represented with the correct Lean4 syntax?  
   - Example: "At least 5" must use `≥ 5`, not `> 5`.  
3. **Boundary Conditions**:  
   - Are implicit constraints (e.g., `n ≥ k` for combinations) explicitly handled or validated?  
4. **Conclusion Consistency**:  
   - Does the Lean4 theorem’s conclusion match the problem’s requested output type?  
   - Example: A value request (e.g., "compute X") must yield an equality (e.g., `= 210`), not a proof of existence.  

**Input Format**:  
- **Original Problem**: [Natural language description]  
- **Lean4 Theorem**: [Formal code statement]  

**Output Requirements**:  
1. First give your thinking and explanations for your judgement.
2. Then give your final judgement separated by ###START### and ended by ###END###, and the judgement could only be YES or NO.

For example, your response should be like this:

Thinking and explanations for your judgement...
###START###
Final judgement: YES|NO
###END###

- **Original Problem**: {problem}
- **Lean4 Theorem**: {theorem}
"""

def main(args: Arguments):
    result_file = f"{args.result_dir}/{args.dataset_name}/{args.model_name_or_path.split('/')[-1]}.json"

    with open(result_file, "r", encoding="utf-8") as f:
        results = json.load(f)
    
    local_start, local_end = utils.alloc_data(args.local_rank, args.num_workers, len(results))
    results = results[local_start:local_end]

    prompts = []
    for result in results:
        problem = result["problem"]
        for fl in result['fl']:
            theorem = fl
            prompt = template.format(problem=problem, theorem=theorem)
            prompts.append(prompt)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=1,
        stop=["###END###"]
    )
    llm = LLM(model=args.judge_model_name_or_path, tensor_parallel_size=args.tensor_parallel_size)

    outputs = llm.generate(prompts, sampling_params)

    for i in range(len(results)):
        results[i]['fl_check_res'] = []
        results[i]['fl_check_res_raw'] = []
        for j in range(len(results[i]['fl'])):
            results[i]['fl_check_res_raw'].append(outputs[i*len(results[i]['fl']) + j].outputs[0].text)
            # fl_check_res = outputs[i*len(results[i]['fl']) + j].outputs[0].text.split("###START###")[-1]
            fl_check_res = results[i]['fl_check_res_raw'][j].split("###START###")[-1]
            if 'yes' in fl_check_res.lower():
                results[i]['fl_check_res'].append(True)
            else:
                results[i]['fl_check_res'].append(False)
            

    with open(f'{args.result_dir}/{args.dataset_name}/{args.model_name_or_path.split("/")[-1]}_{args.local_rank}.json', "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    


if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args()
    main(args)
