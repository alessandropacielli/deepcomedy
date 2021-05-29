import tensorflow as tf
import re


def preprocess_text(
    text,
    token_sep="<SEP>",
    start_of_verse="<GO>",
    end_of_verse="<EOV>",
    end_of_tercet="<EOT>",
    word_level=False,
):

    """
    Accepts text in the form:

    |Nel |mez|zo |del |cam|min |di |no|stra |vi|ta
    |mi |ri|tro|vai |per |u|na |sel|va o|scu|ra,
    |ché |la |di|rit|ta |via |e|ra |smar|ri|ta.

    |Ahi |quan|to a |dir |qual |e|ra è |co|sa |du|ra

    Performs the following:

        1. Translates whitespaces as following:
            · single/multiple spaces --> <SEP> token
            · single newline character --> End-of-Verse (<EOV>)
            · multiple newline characters --> End-of-Tercet (<EOT>)

        2. Adds a <GO> token in the beginning of each verse.
        3. Adds a space between each token (char if word_level is False, words otherwise)
    """

    # Strip each verse
    text = "\n".join([line.strip() for line in text.split("\n")])

    # Add a space after each character (single space becomes double space)
    text = re.sub(r"(.)", r"\1 ", text).strip()

    # Substitute multiple spaces with <SEP>
    text = re.sub(r" {2,}", " {} ".format(token_sep), text)

    # Substitute double newline with End-of-Tercet token
    text = re.sub(
        r"\n{2,}",
        " {} {} {} ".format(end_of_verse, end_of_tercet, start_of_verse),
        text,
    )

    # Substitute single newline with start of verse token
    text = re.sub(r"\n", " {} {} ".format(end_of_verse, start_of_verse), text)

    # Add first GO and last EOT tokens
    text = start_of_verse + " " + text + " " + end_of_verse + " " + end_of_tercet

    # Substitute multiple spaces with single space
    text = re.sub(r" {2,}", " ", text)

    if word_level:
        # Remove spaces
        text = re.sub(r" ", "", text)

        # Add spaces to tags and punctuation
        text = re.sub(r"<[^>]*>", " \g<0> ", text)
        text = re.sub(r'[-:,?“‘\)—»«!”\(";.’]', " \g<0> ", text)

    text = text.strip()

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


def make_dataset(*sequences, batch_size=32):
    buffer_size = len(sequences[0])

    dataset = tf.data.Dataset.from_tensor_slices(tuple(sequences)).shuffle(buffer_size)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset