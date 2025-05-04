GPUS=(0 1)
NUM_GPUS=${#GPUS[@]}
judge_model_name_or_path="Qwen/QwQ-32B"
tensor_parallel_size=1
model_name_or_path="Qwen/Qwen2.5-Math-1.5B-Instruct"
dataset_name="olympiad_bench"
result_dir="results"

NUM_WORKERS=$((NUM_GPUS / tensor_parallel_size))

temperature=0.6
max_tokens=8192


for i in $(seq 0 $((NUM_WORKERS - 1))); do
    CUR_GPUS=$(echo "${GPUS[@]:$((i * tensor_parallel_size)):$tensor_parallel_size}" | tr ' ' ',')
    CUDA_VISIBLE_DEVICES=$CUR_GPUS python scripts/check_trans.py \
        --local_rank $i \
        --num_workers $NUM_WORKERS \
        --judge_model_name_or_path $judge_model_name_or_path \
        --tensor_parallel_size $tensor_parallel_size \
        --model_name_or_path $model_name_or_path \
        --dataset_name $dataset_name \
        --result_dir $result_dir &
done

wait

python scripts/aggregate.py \
    --result_dir $result_dir \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name_or_path \
    --num_files $NUM_WORKERS



