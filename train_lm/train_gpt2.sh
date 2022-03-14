#!/bin/bash

export HF_DATASETS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/datasets/
export TRANSFORMERS_CACHE=/mas/u/hjian42/communityLM/Partisan-LM/cache/huggingface/transformers/

#################################
# GPT-2 (from scratch) -- small 2M data
#################################
# ---> Models
# ./models/gpt2_${which_party}_2M_batch16
# TODO: rename batch8 into batch16

export CUDA_VISIBLE_DEVICES=0
data_folder=usa_tweets_2022
which_party=dems

python ./code/run_clm.py \
    --num_train_epochs 5 \
    --save_steps 50000 \
    --overwrite_output_dir \
    --save_total_limit 10 \
    --logging_first_step \
    --block_size 128 \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --train_file ../data/${data_folder}/${which_party}.train.small.txt \
    --validation_file ../data/${data_folder}/${which_party}.eval.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/gpt2_${which_party}_2M_batch16


#################################
# GPT-2 (pre-trained) -- small 2M data
#################################
# ---> Models
# ./models/pretrained_gpt2_${which_party}_2M_batch16

export CUDA_VISIBLE_DEVICES=3
data_folder=usa_tweets_2022
which_party=reps

python ./code/run_clm.py \
    --num_train_epochs 5 \
    --save_steps 50000 \
    --overwrite_output_dir \
    --save_total_limit 10 \
    --logging_first_step \
    --block_size 128 \
    --model_name_or_path gpt2 \
    --train_file ../data/${data_folder}/${which_party}.train.small.txt \
    --validation_file ../data/${data_folder}/${which_party}.eval.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/pretrained_gpt2_${which_party}_2M_batch16


#################################
# GPT-2 (pre-trained) -- FULL data
#################################
# ---> Models
# ./models/gpt2_${which_party}_full_batch16

export CUDA_VISIBLE_DEVICES=0
data_folder=usa_tweets_2022
which_party=reps
python ./code/run_clm.py \
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
    --train_file ../data/${data_folder}/${which_party}.train.txt \
    --validation_file ../data/${data_folder}/${which_party}.eval.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/gpt2_${which_party}_full_data \
    > ./logs/logs_gpt2_${which_party}_full_data.log 2>&1 &


export CUDA_VISIBLE_DEVICES=1
data_folder=usa_tweets_2022
which_party=dems
python ./code/run_clm.py \
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
    --train_file ../data/${data_folder}/${which_party}.train.23m.txt \
    --validation_file ../data/${data_folder}/${which_party}.eval.txt \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/gpt2_${which_party}_23M_batch24_backup \
    > ./logs/logs_gpt2_${which_party}_23M_batch24_backup.log 2>&1 &


export CUDA_VISIBLE_DEVICES=0
data_folder=usa_tweets_2022
which_party=dems
python ./code/run_clm.py \
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
    --train_file ../data/${data_folder}/${which_party}.train.txt \
    --validation_file ../data/${data_folder}/${which_party}.eval.txt \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/gpt2_${which_party}_100M_batch24_backup \
    > ./logs/logs_gpt2_${which_party}_100M_batch24_backup.log 2>&1 &


#################################
# GPT-2 (from scratch) -- FULL data
#################################
# ---> Models
# ./models/scratch_gpt2_${which_party}_full_batch24
# TODO: run dems

export CUDA_VISIBLE_DEVICES=3
data_folder=usa_dem_rep2022/lm_data_splits
which_party=reps
python ./code/run_clm.py \
    --num_train_epochs 20 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 25 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --logging_first_step \
    --block_size 128 \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --train_file ../data/${data_folder}/${which_party}.train.txt \
    --validation_file ../data/${data_folder}/${which_party}.eval.txt \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/scratch_gpt2_${which_party}_full_batch24 \
    > ./logs/logs_scratch_gpt2_${which_party}_full_batch24.log 2>&1 &


export CUDA_VISIBLE_DEVICES=2
data_folder=usa_dem_rep2022/lm_data_splits
which_party=dems
python ./code/run_clm.py \
   --num_train_epochs 20 \
    --logging_strategy epoch \
    --save_strategy epoch \
    --save_total_limit 25 \
    --evaluation_strategy epoch \
    --overwrite_output_dir \
    --logging_first_step \
    --block_size 128 \
    --model_type gpt2 \
    --tokenizer_name gpt2 \
    --train_file ../data/${data_folder}/${which_party}.train.23m.txt \
    --validation_file ../data/${data_folder}/${which_party}.eval.txt \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/scratch_gpt2_${which_party}_23M_batch24 \
    > ./logs/logs_scratch_gpt2_${which_party}_23M_batch24.log 2>&1 &


export CUDA_VISIBLE_DEVICES=2
data_folder=usa_dem_rep2022/lm_data_splits
which_party=dems
python ./code/run_clm.py \
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
    --train_file ../data/${data_folder}/${which_party}.train.txt \
    --validation_file ../data/${data_folder}/${which_party}.eval.txt \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir ./models/scratch_gpt2_${which_party}_100M_batch24 \
    > ./logs/logs_scratch_gpt2_${which_party}_100M_batch24.log 2>&1 &
