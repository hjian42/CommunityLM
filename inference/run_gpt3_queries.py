#!/usr/bin/env python3

"""
Generates the GPT-3 output for the CommunityLM paper
This program depends on https://github.com/social-machines/sounding-board/
See constants below for important settings
"""

import json
import os
import pandas as pd
import streamlit_soundingboard as ss

INPUT_FILE = "anes2020_pilot_prompt_probing.csv"

# This number, multiplied by ss.PANEL_SIZE (see below), should be the total
# number of instances to generate per prompt (1000 for the CommunityLM experiments)
NUM_ITERATIONS = 10

# File to save all the raw GPT3 output for safekeeping
GPT3_OUTPUT_FILENAME = "results.prompted-davinci"

# Where to write the text files used as input for sentiment scoring
TEXT_OUTPUT_BASE = "../data/"

# Set below to false if you've already run the GPT3 queries and just want to re-parse
# the results and write them out to TEXT_OUTPUT_BASE
RERUN_GPT3 = True


def parse_results(results):
    for res in results:
        prompt_version = res["prompt_version"]
        iteration = res["iteration"]
        pid = res["pid"]
        for community, res2 in res["result"].items():
            for c in ss.COMMUNITIES:
                if community == c["display_name"]:
                    community = c["short_name"]
                    break
            prefix = res["input_row"][prompt_version]
            if not res2[0]:
                print(iteration, pid, community, prompt_version)
                continue
            text = "\n".join(
                prefix + " " + choice["text"].strip().split("\n")[0]
                for choice in res2[0]["choices"]
            )
            file_dir = os.path.join(
                TEXT_OUTPUT_BASE, community, "run_" + str(iteration + 1), prompt_version
            )
            os.makedirs(file_dir, exist_ok=True)
            fn = os.path.join(file_dir, pid + ".txt")
            with open(fn, "a") as fs_out:
                print(text, file=fs_out)


def aggregate_opinion(prompt, text_gen, SentimentModel):
    Sentiment2score = {"Negative": 0, "Positive": 100, "Neutral": 50}
    results = generate_with_a_prompt(prompt, text_gen)
    preds = SentimentModel(results)
    scores = [Sentiment2score[pred["label"]] for pred in preds]
    return np.array(scores).mean(), preds


def parse_results_helper():
    df = pd.read_csv(INPUT_FILE, header=None)
    pred_labels = []
    pred_avg_sentiments = []
    gold_labels = []
    prompts = []
    for prompt, gold_label in df[["Prompt4", "is_repub_leading"]].values:
        prompt = prompt.strip()
        gold_label = int(gold_label)
        # gold_label = {"positive": 1, "negative": 0}[gold_label]

        avg_dem_sentiment, _ = aggregate_opinion(prompt, dem_generator, sentiment_model)
        avg_repub_sentiment, _ = aggregate_opinion(
            prompt, repub_generator, sentiment_model
        )

        pred_avg_sentiments.append((avg_repub_sentiment, avg_dem_sentiment))
        if avg_repub_sentiment > avg_dem_sentiment:
            pred_label = 1
        else:
            pred_label = 0

        pred_labels.append(pred_label)
        gold_labels.append(gold_label)
        prompts.append(prompt)
        print(prompt)
        print("Gold Label:", gold_label)
        print("Pred Label:", pred_label)
        print()


def generate_gpt3_completions(iteration=0):
    prompts = pd.read_csv(INPUT_FILE)
    fs_out = open(GPT3_OUTPUT_FILENAME, "w")

    results = []
    for iteration in range(NUM_ITERATIONS):
        for index, row in prompts.iterrows():  # each row corresponds to a public figure
            for prompt_version in ["Prompt1", "Prompt2", "Prompt3", "Prompt4"]:
                prompt = "As a ${party_affiliation}, I think " + row[prompt_version]
                # add space?
                res = ss.get_results_for_section(
                    {"question": "", "answer_type": "open-ended"}, prompt
                )
                res = dict(res)
                x = {
                    "prompt": prompt,
                    "prompt_version": prompt_version,
                    "iteration": iteration,
                    "pid": row["pid"],
                    "input_row": dict(row),
                    "result": res,
                }
                results.append(x)
                print(json.dumps(x), file=fs_out)
    return results


if __name__ == "__main__":
    ss.OPEN_ENDED_TEMPERATURE = 1.0
    # Note that the GPT3 API supports max of 128 for "n", the number of
    # completions to generate.  To generate 1000, we need to run 10 iterations
    # (NUM_ITERATIONS above) of 100 completions (PANEL_SIZE below).
    ss.PANEL_SIZE = 100
    ss.COMMUNITIES = [
        {
            "display_name": "Democrat GPT-3",
            "short_name": "democrat-gpt3",
            "party_affiliation": "Democrat",
            "model_type": "gpt3",
            # These fine-tune ids were ignored for the paper experiments, for which
            # we used the base curie model
            "model_id": "curie:ft-center-for-constructive-communication-2022-05-06-21-59-56",
        },
        {
            "display_name": "Republican GPT-3",
            "short_name": "republican-gpt3",
            "party_affiliation": "Republican",
            "model_type": "gpt3",
            # These fine-tune ids were ignored for the paper experiments, for which
            # we used the base curie model
            "model_id": "curie:ft-center-for-constructive-communication-2022-05-06-18-22-42",
        },
    ]

    results = []
    if RERUN_GPT3:
        for i in range(10):
            results = generate_gpt3_completions(iteration=i)
    else:
        results = []
        with open(GPT3_OUTPUT_FILENAME) as fs:
            for line in fs:
                results.append(json.loads(line))

    parse_results(results)
