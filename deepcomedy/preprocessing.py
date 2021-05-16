import tensorflow as tf


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


# TODO implement ngrams