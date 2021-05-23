import tensorflow as tf
import re


def is_empty(str):
    return str == ""


def is_not_empty(str):
    return str != ""


def strip(x):
    return x.strip()


def preprocess_tercets(text):

    # Drop the first syllable separator from the syllabified text
    text = "\n".join([line.strip() for line in text.split("\n")])

    # Add a space after each character (single space becomes double space)
    text = re.sub(r"(.)", r"\1 ", text).strip()

    # Substitute multiple spaces with <SEP>
    text = re.sub(r" {2,}", " <SEP> ", text)

    # Substitute double newline with End-of-Tercet token
    text = re.sub(r"\n{2,}", " <EOT> <GO> ", text)

    # Substitute single newline with start of verse token
    text = re.sub(r"\n", " <GO> ", text)

    # Substitute multiple spaces with single space
    text = re.sub(r" {2,}", " ", text)

    # Add first GO and last EOT tokens
    text = "<GO> " + text + " <EOT>"

    return text


def slice_windows(text, window_size):
    text_windows = []

    for i in range(0, len(text) - window_size + 1, window_size):
        text_windows.append(text[i : i + window_size])

    return tf.convert_to_tensor(text_windows)


def load_verses(path, char_level=False, pad=False, tokenize=True):
    """
    Loads verses from path, encodes them using a tokenizer and pads them so they all have the same dimension.
    """
    raw_text = open(path, "rb").read().decode(encoding="utf-8")

    # TODO improve verse start/end symbol
    # TODO consider tercet end symbol

    if char_level:
        start_symbol = "^"
        end_symbol = "$"
    else:
        start_symbol = "^ "
        end_symbol = " $"

    # Prepend start symbol and append end symbol to each verse
    verses = [
        start_symbol + line.strip() + end_symbol
        for line in raw_text.split("\n")
        if line.strip() != ""
    ]

    if tokenize:
        # Tokenize at char/word level according to input param
        tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters="", char_level=char_level, lower=False
        )
        tokenizer.fit_on_texts(verses)

        result = tokenizer.texts_to_sequences(verses)

        # Pad each verse to the length of the longest verse
        if pad:
            result = tf.keras.preprocessing.sequence.pad_sequences(
                result, padding="post"
            )

        return raw_text, result, tokenizer

    else:
        return raw_text, verses
