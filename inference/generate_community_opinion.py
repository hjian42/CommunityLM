"""
Author: hjian42@icloud.com

This script uses communuty GPT models to generate opinions given prompts and save these voices

>>> Examples:
    # generate 1000 voices for Democratic GPT, repeat 5 times
    export CUDA_VISIBLE_DEVICES=1
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

    # generate 1000 voices for Republican GPT, repeat 5 times
    export CUDA_VISIBLE_DEVICES=2
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
"""

import sys
import contextlib
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed
import os
import torch
import math
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def generate_with_a_prompt(prompt, text_gen_pipeline):
    """
    Generate a list of statements given the prompt based on one GPT-2 model
    
    NOTE: 50256 corresponds to '<|endoftext|>'
    """
    
    results = text_gen_pipeline(prompt, 
                                max_length=50,
                                temperature=0.5,
                                num_return_sequences=100, # 1000 leads to OOM
                                pad_token_id=50256,
                                clean_up_tokenization_spaces=True
                               )
    
    # only use the first utterance
    results = [res['generated_text'].split("\n")[0] for res in results]
    return results

def main():
    
    parser = argparse.ArgumentParser(description="main training script for word2vec dynamic word embeddings...")
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--prompt_data_path", type=str)
    parser.add_argument("--prompt_option", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    set_seed(args.seed)

    model_name = args.model_path.strip("/").split("/")[-1]
    df = pd.read_csv(args.prompt_data_path)
    questions = df.pid.values.tolist()
    prompts = df[args.prompt_option].values.tolist()
    text_generator = pipeline('text-generation', model=args.model_path, device=0)

    output_folder = os.path.join(args.output_path, args.prompt_option)
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for question, prompt in zip(questions, prompts):
        responses = []
        print("Working on [{}]...".format(question))
        for _ in range(10):
            batch_responses = generate_with_a_prompt(prompt, text_generator)
            responses.extend(batch_responses)
        with open(os.path.join(output_folder, question+".txt"), "w") as out:
            for line in responses:
                line = line.replace("\n", " ")
                out.write(line)
                out.write("\n")


if __name__ == "__main__":
    main()