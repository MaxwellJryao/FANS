GPUS=(0 1 2 3 4 5 6 7)
NUM_GPUS=${#GPUS[@]}
model_name_or_path="Qwen/Qwen2.5-Math-7B-Instruct"
dataset_name="olympiad_bench"
dataset_path="FlippyDora/${dataset_name}"

for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python scripts/generate.py \
        --local_rank $i \
        --num_workers $NUM_GPUS \
        --dataset_path $dataset_path \
        --dataset_name $dataset_name \
        --dataset_split "train" \
        --dataset_end 1000 \
        --model_name_or_path $model_name_or_path \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.8 \
        --max_new_tokens 4096 \
        --temperature 0.7 \
        --top_p 0.95 \
        --n 8 &
done

wait

python scripts/aggregate.py \
    --result_dir "results" \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name_or_path \
    --num_files $NUM_GPUS

python scripts/score.py \
    --result_dir "results" \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name_or_path
