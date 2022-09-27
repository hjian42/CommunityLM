# CommunityLM: Probing Partisan Worldviews from Language Models
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-2201.07281-b31b1b.svg)](https://arxiv.org/abs/2209.07065)

This repo contains the code for fine-tuning and evaluating Republican and Democrat Community GPT-2 models. We also release the two models on [HuggingFace Model Hub](https://huggingface.co/CommunityLM), which are fine-tuned on 4.7M (~100M tokens) tweets of Republican Twitter users between 2019-01-01 and 2020-04-10. Details are described in **[CommunityLM: Probing Partisan Worldviews from Language Models](https://arxiv.org/abs/2209.07065)**.


## References

If you use this repository in your research, please kindly cite [our paper](https://arxiv.org/abs/2209.07065): 

```bibtex
@inproceedings{jiang-etal-2022-communitylm,
    title = "CommunityLM: Probing Partisan Worldviews from Language Models",
     author = {Jiang, Hang and Beeferman, Doug and Roy, Brandon and Roy, Deb},
    booktitle = "Proceedings of the 29th International Conference on Computational Linguistics",
    year = "2022",
    publisher = "International Committee on Computational Linguistics",
}
```

## Installation

```bash
pip install git+https://github.com/huggingface/transformers
pip install -r train_lm/requirements.txt
```

## How to use the left and right CommunityLM models from HuggingFace

See more on our [HuggingFace Model Hub Page](https://huggingface.co/CommunityLM).

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("CommunityLM/republican-twitter-gpt2")

model = AutoModelForCausalLM.from_pretrained("CommunityLM/republican-twitter-gpt2")
```

## CommunityLM Framework

### Training

Check `./train_lm/train_gpt2.sh` for fine-tuning GPT-2 on commumity data. 

### Inference

Check `inference/evaluate_community_models.sh` for generating community voices and aggregating the stance. 

### Evaluation

Check `inference/notebooks/evaluate_communitylm.ipynb` for evaluating the model from predictions given in the inference step. This notebook also contains the code to reproduce the ranking plots.

## Acknowledgement

CommunityLM is a research program from MIT Center for Constructive Communication (@mit-ccc) and MIT Media Lab. We are devoted to developing socially-aware language models for community understanding and constructive dialogue. This repository is mainly contributed by [Hang Jiang](https://www.mit.edu/~hjian42/) (@hjian42). 

