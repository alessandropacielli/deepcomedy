import re


def strip_tokens(x):

    x = re.sub(r"[ ]+", "", x)
    x = re.sub("<GO>", "\n", x)
    x = re.sub("<SEP>", " ", x)
    x = re.sub("<[^>]*>", "", x)
    x = x.strip()

    return x


def remove_syll_token(x):
    x = re.sub(r"\|", "", x)

    # Convert multiple spaces to one
    x = re.sub(r"[ ]+", " ", x)

    return x


def remove_punctuation(x):
    x = re.sub('[-:,?“‘\)—»«!”\(";.’]', "", x)
    return x


def is_empty(str):
    return str == ""


def is_not_empty(str):
    return str != ""


def strip(x):
    return x.strip()