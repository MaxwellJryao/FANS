cd ../kimina-lean-server/

python verify_fl.py \
    --result_dir ../FANS/results \
    --dataset_name amc23 \
    --model_name_or_path Qwen/Qwen2.5-Math-1.5B-Instruct \
    --lean_server_host http://localhost \
    --lean_server_port 12332
