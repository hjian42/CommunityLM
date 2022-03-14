from emoji import demojize
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
# from pandarallel import pandarallel
import pandas as pd

tokenizer = TweetTokenizer()
# pandarallel.initialize()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    if lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tweet = tweet.replace('"', "")
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))

    # filter out tweets whose length < 10
    if len(tokens) < 10:
        return ""

    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    return " ".join(normTweet.split())


if __name__ == "__main__":

    with open("./usa_tweets_2022/raw/reps.csv") as f, \
        open("./usa_tweets_2022/normalize/reps_normalize.txt", "w") as out:
        for i, line in tqdm(enumerate(f)):
            if i != 0:
                line = normalizeTweet(line)
                if line.strip():
                    out.write(line)
                    out.write("\n")

    with open("./usa_tweets_2022/raw/dems.csv") as f, \
        open("./usa_tweets_2022/normalize/dems_normalize.txt", "w") as out:
        for i, line in tqdm(enumerate(f)):
            if i != 0:
                line = normalizeTweet(line)
                if line.strip():
                    out.write(line)
                    out.write("\n")