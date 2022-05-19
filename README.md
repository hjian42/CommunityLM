# CommunityLM
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

CommunityLM: Probing Partisan Worldviews from Language Models

1. use `./data/pull_decahose_data.py` and `./data/tweet_process.py` for pulling and pre-processing Tweets
2. check `./train_lm/train_gpt2.sh` for fine-tuning / training GPT-2 on commumity data
3. check `./inference/evaluate_community_models.sh` for evaluation
  - `./inference/anes2020_pilot.ipynb`: evaluate the CommunityLM models
  - `./inference/baselines.ipynb`: evaluate the baseline models
  - `./inference/gpt2-demo.ipynb`: demo code to use GPT-2 for different tasks
