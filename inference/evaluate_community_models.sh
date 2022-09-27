##############################
## finetuned_gpt2_2019_dem ##
##############################
export CUDA_VISIBLE_DEVICES=0
for run in 1 2 3 4 5
do
    for prompt in Prompt1 Prompt2 Prompt3 Prompt4
    do
        python generate_community_opinion.py \
        --model_path ../train_lm/models/finetuned_gpt2_2019_dem/ \
        --prompt_data_path ./notebooks/anes2020_pilot_prompt_probing.csv \
        --prompt_option ${prompt} \
        --output_path ../output/finetuned_gpt2_2019_dem/run_${run} \
        --seed ${run}
    done
done

python compute_group_stance.py \
    --data_folder ../output/finetuned_gpt2_2019_dem \
    --anes_csv_file ./notebooks/anes2020_pilot_prompt_probing.csv \
    --output_filename ../output/finetuned_gpt2_2019_dem/finetuned_gpt2_group_stance_predictions.csv

################################
## finetuned_gpt2_2019_repub ##
################################
export CUDA_VISIBLE_DEVICES=1
for run in 1 2 3 4 5
do
    for prompt in Prompt1 Prompt2 Prompt3 Prompt4
    do
        python generate_community_opinion.py \
        --model_path ../train_lm/models/finetuned_gpt2_2019_repub/ \
        --prompt_data_path ./notebooks/anes2020_pilot_prompt_probing.csv \
        --prompt_option ${prompt} \
        --output_path ../output/finetuned_gpt2_2019_repub/run_${run} \
        --seed ${run}
    done
done

python compute_group_stance.py \
    --data_folder ../output/finetuned_gpt2_2019_repub \
    --anes_csv_file ./notebooks/anes2020_pilot_prompt_probing.csv \
    --output_filename ../output/finetuned_gpt2_2019_repub/finetuned_gpt2_group_stance_predictions.csv
