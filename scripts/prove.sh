GPUS=(0 1 2 4 5 6 7 8 9)
NUM_GPUS=${#GPUS[@]}

prover_model_name_or_path="deepseek-ai/DeepSeek-Prover-V2-7B"
tensor_parallel_size=1
gpu_memory_utilization=0.8

NUM_WORKERS=$((NUM_GPUS / tensor_parallel_size))

max_new_tokens=8192
max_model_len=16384

result_dir="results"
dataset_name="minerva_math"
model_name_or_path="Qwen/Qwen2.5-Math-7B-Instruct"

seed=30

# use 2 GPUs per worker
for i in $(seq 0 $((NUM_WORKERS - 1))); do
    CUR_GPUS=$(echo "${GPUS[@]:$((i * tensor_parallel_size)):$tensor_parallel_size}" | tr ' ' ',')
    CUDA_VISIBLE_DEVICES=$CUR_GPUS python scripts/prove.py \
        --local_rank $i \
        --num_workers $NUM_WORKERS \
        --result_dir $result_dir \
        --dataset_name $dataset_name \
        --model_name_or_path $model_name_or_path \
        --prover_model_name_or_path $prover_model_name_or_path \
        --tensor_parallel_size $tensor_parallel_size \
        --gpu_memory_utilization $gpu_memory_utilization \
        --max_new_tokens $max_new_tokens \
        --max_model_len $max_model_len \
        --seed $seed &
done

wait

python scripts/aggregate.py \
    --result_dir $result_dir \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name_or_path \
    --num_files $NUM_WORKERS