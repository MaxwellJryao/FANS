import sys
import nest_asyncio
import uuid
import json
from transformers import HfArgumentParser
from dataclasses import dataclass, field

from client.client import Lean4Client
from client.infotree import extract_data
from utils.proof_utils import split_proof_header, parse_client_response

nest_asyncio.apply()

@dataclass
class Arguments:
    lean_server_host: str = field(default="http://localhost")
    lean_server_port: int = field(default=12332)
    result_dir: str = field(default="../FANS/results")
    dataset_name: str = field(default="amc23")
    model_name_or_path: str = field(default="Qwen/Qwen2.5-Math-1.5B-Instruct")

formal_statement_template = """
import Mathlib
import Aesop

set_option maxHeartbeats 0

open BigOperators Real Nat Topology Rat

{fl_statement}
""".strip()

def main(args: Arguments):
    client = Lean4Client(base_url=f"{args.lean_server_host}:{args.lean_server_port}")

    results_file = f"{args.result_dir}/{args.dataset_name}/{args.model_name_or_path.split('/')[-1]}.json"
    with open(results_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    codes = []
    custom_id = 0
    for result in results:
        for proof in result["proofs"]:
            code = formal_statement_template.format(fl_statement=proof)
            codes.append({
                "proof": code,
                "custom_id": str(custom_id)
            })
            custom_id += 1

    response = client.verify(codes, timeout=60)

    verification_results = [
        parse_client_response(item)
        for item in response["results"]
    ]

    # verify resulst
    # has_error, is_valid_no_sorry, is_valid_with_sorry, time

    for i, result in enumerate(results):
        verify_results = []
        for j in range(len(result["proofs"])):
            verify_results.append(verification_results[i * len(result["proofs"]) + j])

        result["fl_verify_results"] = verify_results
    
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = HfArgumentParser(Arguments)
    args = parser.parse_args()
    main(args)
    