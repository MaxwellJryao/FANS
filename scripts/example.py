from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from vllm import LLM, SamplingParams
torch.manual_seed(30)

model_id = "deepseek-ai/DeepSeek-Prover-V2-7B"  # or DeepSeek-Prover-V2-671B
tokenizer = AutoTokenizer.from_pretrained(model_id)

formal_statement = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

/-- What is the positive difference between $120\%$ of 30 and $130\%$ of 20? Show that it is 10.-/
theorem mathd_algebra_10 : abs ((120 : ℝ) / 100 * 30 - 130 / 100 * 20) = 10 := by
  sorry
""".strip()

prompt = """
Complete the following Lean 4 code:

```lean4
{}
```

Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies.
The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
""".strip()

chat = [
  {"role": "user", "content": prompt.format(formal_statement)},
]

# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model = LLM(model_id, tensor_parallel_size=2, gpu_memory_utilization=0.8)
inputs = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

sampling_params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192)
import time
start = time.time()
outputs = model.generate([inputs], sampling_params=sampling_params)
print(outputs[0].outputs[0].text)
print(time.time() - start)
