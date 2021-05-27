def strip_tokens(x):

    x = re.sub(r"[ ]+", "", x)
    x = re.sub("<GO>", "\n", x)
    x = re.sub("<SEP>", " ", x)
    x = re.sub("<[^>]*>", "", x)

    return x


def remove_syll_token(x):
    x = re.sub(r"\|", "", x)

    # Convert multiple spaces to one
    x = re.sub(r"[ ]+", " ", x)

    return x


def remove_punctuation(x):
    x = re.sub('[-:,?“‘\)—»«!”\(";.’]', "", x)
    return x