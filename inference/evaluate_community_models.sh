##############################
## pretrained_gpt2_2019_dem ##
##############################
export CUDA_VISIBLE_DEVICES=0
for run in 1 2 3 4 5
do
    for prompt in Prompt1 Prompt2 Prompt3 Prompt4
    do
        python generate_community_opinion.py \
        --model_path ../train_lm/models/pretrained_gpt2_2019_dem/ \
        --prompt_data_path ./anes2020_pilot_prompt_probing.csv \
        --prompt_option ${prompt} \
        --output_path ../output/pretrained_gpt2_2019_dem/run_${run} \
        --seed ${run}
    done
done

python compute_group_stance.py \
    --data_folder ../output/pretrained_gpt2_2019_dem \
    --anes_csv_file ./anes2020_pilot_prompt_probing.csv \
    --output_filename ../output/pretrained_gpt2_2019_dem/group_stance_predictions.csv

################################
## pretrained_gpt2_2019_repub ##
################################
export CUDA_VISIBLE_DEVICES=1
for run in 1 2 3 4 5
do
    for prompt in Prompt1 Prompt2 Prompt3 Prompt4
    do
        python generate_community_opinion.py \
        --model_path ../train_lm/models/pretrained_gpt2_2019_repub/ \
        --prompt_data_path ./anes2020_pilot_prompt_probing.csv \
        --prompt_option ${prompt} \
        --output_path ../output/pretrained_gpt2_2019_repub/run_${run} \
        --seed ${run}
    done
done

python compute_group_stance.py \
    --data_folder ../output/pretrained_gpt2_2019_repub \
    --anes_csv_file ./anes2020_pilot_prompt_probing.csv \
    --output_filename ../output/pretrained_gpt2_2019_repub/group_stance_predictions.csv

############################
## scratch_gpt2_2019_dem  ##
############################
export CUDA_VISIBLE_DEVICES=2
for run in 1 2 3 4 5
do
    for prompt in Prompt1 Prompt2 Prompt3 Prompt4
    do
        python generate_community_opinion.py \
        --model_path ../train_lm/models/scratch_gpt2_2019_dem/ \
        --prompt_data_path ./anes2020_pilot_prompt_probing.csv \
        --prompt_option ${prompt} \
        --output_path ../output/scratch_gpt2_2019_dem/run_${run} \
        --seed ${run}
    done
done

python compute_group_stance.py \
    --data_folder ../output/scratch_gpt2_2019_dem \
    --anes_csv_file ./anes2020_pilot_prompt_probing.csv \
    --output_filename ../output/scratch_gpt2_2019_dem/group_stance_predictions.csv

##############################
## scratch_gpt2_2019_repub  ##
##############################
export CUDA_VISIBLE_DEVICES=3
for run in 1 2 3 4 5
do
    for prompt in Prompt1 Prompt2 Prompt3 Prompt4
    do
        python generate_community_opinion.py \
        --model_path ../train_lm/models/scratch_gpt2_2019_repub/ \
        --prompt_data_path ./anes2020_pilot_prompt_probing.csv \
        --prompt_option ${prompt} \
        --output_path ../output/scratch_gpt2_2019_repub/run_${run} \
        --seed ${run}
    done
done

python compute_group_stance.py \
    --data_folder ../output/scratch_gpt2_2019_repub \
    --anes_csv_file ./anes2020_pilot_prompt_probing.csv \
    --output_filename ../output/scratch_gpt2_2019_repub/group_stance_predictions.csv
