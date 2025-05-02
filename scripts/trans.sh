GPUS=(0 1 2 3 4 5 6 9)
NUM_GPUS=${#GPUS[@]}
translate_model_name_or_path="AI-MO/Kimina-Autoformalizer-7B"
dataset_name="aime24"
model_name_or_path="Qwen/Qwen2.5-Math-1.5B-Instruct"
result_dir="results"

for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} python scripts/translate.py \
        --local_rank $i \
        --num_workers $NUM_GPUS \
        --result_dir $result_dir \
        --dataset_name $dataset_name \
        --translate_model_name_or_path $translate_model_name_or_path \
        --model_name_or_path $model_name_or_path &
done

wait

python scripts/aggregate.py \
    --result_dir $result_dir \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name_or_path \
    --num_files $NUM_GPUS