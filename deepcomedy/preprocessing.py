import tensorflow as tf


def load_verses(path, char_level=False):
    """
    Loads verses from path, encodes them using a tokenizer and pads them so they all have the same dimension.
    """
    raw_text = open(input_file, "rb").read().decode(encoding="utf-8")

    # TODO improve verse start/end symbol
    # TODO consider tercet end symbol

    if char_level:
        start_symbol = "^"
        end_symbol = "$"
    else:
        start_symbol = "^ "
        end_symbol = " $"

    verses = [
        start_symbol + line.strip() + end_symbol
        for line in raw_text.split("\n")
        if line.strip() != ""
    ]

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters="", char_level=char_level, lower=False
    )
    tokenizer.fit_on_texts(raw_text)

    raw_text_encoded = tokenizer.texts_to_sequences(raw_text)

    padded_text = tf.keras.preprocessing.sequence.pad_sequences(
        raw_text_encoded, padding="post"
    )

    return padded_text


# TODO implement ngrams