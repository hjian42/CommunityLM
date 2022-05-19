#!/bin/bash
# pip install git+https://github.com/huggingface/transformers
# export PATH="/opt/conda/bin:$PATH"

export HF_DATASETS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/datasets/
export TRANSFORMERS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/transformers/


#################################
# GPT-2 (pre-trained) -- partisan data
#################################

export CUDA_VISIBLE_DEVICES=0
data_folder=usa_tweets_2019
which_party=repub
python ./run_clm.py \
    --num_train_epochs 10 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 25 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --logging_first_step \
    --block_size 128 \
    --model_name_or_path gpt2 \
    --tokenizer_name gpt2 \
    --train_file ../data/${data_folder}/${which_party}_4.7M_tweets_proc.txt \
    --validation_split_percentage 2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/pretrained_gpt2_2019_${which_party}
    # > ./logs/logs_0504_gpt2_${which_party}_full_data.log 2>&1 &

export CUDA_VISIBLE_DEVICES=1
data_folder=usa_tweets_2019
which_party=dem
python ./run_clm.py \
    --num_train_epochs 10 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 25 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --logging_first_step \
    --block_size 128 \
    --model_name_or_path gpt2 \
    --tokenizer_name gpt2 \
    --train_file ../data/${data_folder}/${which_party}_4.7M_tweets_proc.txt \
    --validation_split_percentage 2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/pretrained_gpt2_2019_${which_party}


#################################
# GPT-2 (pre-trained) -- FULL data
#################################
export HF_DATASETS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/datasets/
export TRANSFORMERS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/transformers/

export CUDA_VISIBLE_DEVICES=2
data_folder=usa_tweets_2019
python ./run_clm.py \
    --num_train_epochs 10 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 25 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --logging_first_step \
    --block_size 128 \
    --model_name_or_path gpt2 \
    --tokenizer_name gpt2 \
    --train_file ../data/${data_folder}/combined_dem_rep_9.4M_proc.txt \
    --validation_split_percentage 2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/pretrained_gpt2_2019_combined_dem_rep


#################################
# GPT-2 (from scratch) -- partisan data
#################################
export HF_DATASETS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/datasets/
export TRANSFORMERS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/transformers/
export CUDA_VISIBLE_DEVICES=2
data_folder=usa_tweets_2019
which_party=repub
python ./run_clm.py \
    --num_train_epochs 10 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 25 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --logging_first_step \
    --block_size 128 \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --train_file ../data/${data_folder}/${which_party}_4.7M_tweets_proc.txt \
    --validation_split_percentage 2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/scratch_gpt2_2019_${which_party}



export CUDA_VISIBLE_DEVICES=3
which_party=dem
data_folder=usa_tweets_2019
python ./run_clm.py \
    --num_train_epochs 10 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 25 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --logging_first_step \
    --block_size 128 \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --train_file ../data/${data_folder}/${which_party}_4.7M_tweets_proc.txt \
    --validation_split_percentage 2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/scratch_gpt2_2019_${which_party}


#################################
# GPT-2 (from scratch) -- FULL data
#################################
export HF_DATASETS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/datasets/
export TRANSFORMERS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/transformers/
export CUDA_VISIBLE_DEVICES=3
data_folder=usa_tweets_2019
python ./run_clm.py \
    --num_train_epochs 10 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 25 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --logging_first_step \
    --block_size 128 \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --train_file ../data/${data_folder}/combined_dem_rep_9.4M_proc.txt \
    --validation_split_percentage 2 \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/scratch_gpt2_2019_combined_dem_rep
