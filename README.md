# FANS - Formal Answer Selection Based on Lean4
## Environment Setup
1. Config the lean4 verifier environment according to [Kimina Lean Server](https://github.com/project-numina/kimina-lean-server).
2. Create a new python environment.
```bash
python -m venv ~/.python/fl
source ~/.python/fl/bin/activate
```
3. Clone this repository
```bash
git clone https://github.com/MaxwellJryao/FANS.git
```
4. Install dependent packages.
```bash
python -m uv pip install vllm math-verify
```
5. Copy `scripts/verify_fl.py` to the folder of `kimina-lean-server`.
6. Run the experiments.
    1. Generate new responses.
    2. Score the responses.
    3. Translate NL into FL.
    4. Prove for FL.
    5. Verify FL.