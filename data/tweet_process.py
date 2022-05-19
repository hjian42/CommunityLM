from emoji import demojize
from nltk.tokenize import TweetTokenizer
from tqdm import tqdm
# from pandarallel import pandarallel
import pandas as pd
import sys

tokenizer = TweetTokenizer()
# pandarallel.initialize()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    if lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        # return "HTTPURL"
        return ""
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
    
    intput_file = sys.argv[1]
    output_file = sys.argv[2]
    print(intput_file, output_file)

    with open(intput_file) as f, \
        open(output_file, "w") as out:
        for i, line in tqdm(enumerate(f)):
            line = normalizeTweet(line)
            if line.strip():
                out.write(line)
                out.write("\n")
                