"""
Author: hjian42@icloud.com

This script computes the average group stance for one community GPT model per run and prompt

>> Examples:

    # generate the group sentiment for the Democrat GPT model
    export CUDA_VISIBLE_DEVICES=1
    python compute_group_stance.py \
    --data_folder ../output/pretrained_gpt2_2019_dem \
    --anes_csv_file ./anes2020_pilot_prompt_probing.csv \
    --output_filename ../output/pretrained_gpt2_2019_dem/group_stance_predictions.csv

    # generate the group sentiment for the Republican GPT model
    export CUDA_VISIBLE_DEVICES=2
    python compute_group_stance.py \
    --data_folder ../output/pretrained_gpt2_2019_repub \
    --anes_csv_file ./anes2020_pilot_prompt_probing.csv \
    --output_filename ../output/pretrained_gpt2_2019_repub/group_stance_predictions.csv

"""

from transformers import pipeline, set_seed
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import pandas as pd
import os
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TextDataset(Dataset):
    def __init__(self, text_file_path):
        with open(text_file_path) as f:
            self.texts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def compute_group_sentiment(filepath, model_pipeline):
    """
    by default, the model_pipeline is a neural sentiment model
    """
    sentiment_dict = {
        "Negative": 0,
        "Positive": 100,
        "Neutral": 50
    }
    
    dataset = TextDataset(filepath)
    dataloader = DataLoader(dataset, batch_size=400, shuffle=False)
    
    all_scores = []
    for batch in dataloader:
        preds = model_pipeline(batch)
        scores = [sentiment_dict[pred['label']] for pred in preds]
        all_scores.extend(scores)
    group_sentiment = np.array(all_scores).mean()
    
    return group_sentiment


def compute_group_lexicon_sentiment(filepath, lexicon_model):
    """
    lexicon-based sentiment model
    """
    with open(filepath) as f:
        texts = [line.strip() for line in f.readlines()]
    all_scores = []
    for sentence in texts:
        score = lexicon_model.polarity_scores(sentence)['compound']
        all_scores.append(score)
    group_sentiment = np.array(all_scores).mean()
    return group_sentiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str)
    parser.add_argument("--anes_csv_file", type=str)
    parser.add_argument("--output_filename", type=str)
    parser.add_argument("--sentiment_model_type", default="neural", type=str)
    parser.add_argument("--framework", default="gpt", type=str)
    args = parser.parse_args()

    questions = pd.read_csv(args.anes_csv_file).pid.values.tolist()
    model_name = args.data_folder.strip("/").split("/")[-1]

    if args.sentiment_model_type == "neural":
        sentiment_pipeline = pipeline("sentiment-analysis",
                            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                            tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                            max_length=512, 
                            truncation=True,
                            device=0)
    else: # lexicon
        sentiment_pipeline = SentimentIntensityAnalyzer()
    
    if args.framework == "gpt":
        columns = ['model_name', 'run', 'prompt_format', 'question', 'group_sentiment']
        rows = []
        for run in range(1, 6):
            run = "run_{}".format(run)
            print("Processing {} ...".format(run))
            for prompt_format in range(1, 5):
                prompt_format = "Prompt{}".format(prompt_format)
                for question in questions:
                    file_name = os.path.join(args.data_folder, run, prompt_format, question+".txt")
                    if args.sentiment_model_type == "neural":
                        group_sentiment = compute_group_sentiment(file_name, sentiment_pipeline)
                    else:
                        group_sentiment = compute_group_lexicon_sentiment(file_name, sentiment_pipeline)
                    rows.append([model_name, run, prompt_format, question, group_sentiment])

        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(args.output_filename)
    else: #  "keyword" baseline
        rows = []
        columns = ['model_name', 'question', 'group_sentiment']
        for question in questions:
            print("Processing {} ...".format(question))
            file_name = os.path.join(args.data_folder, question+".txt")
            if args.sentiment_model_type == "neural":
                group_sentiment = compute_group_sentiment(file_name, sentiment_pipeline)
            else:
                group_sentiment = compute_group_lexicon_sentiment(file_name, sentiment_pipeline)
            rows.append([model_name, question, group_sentiment])
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(args.output_filename)

if __name__ == "__main__":
    main()