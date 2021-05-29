from .preprocessing import is_empty, is_not_empty, strip
import tensorflow_datasets as tfds


def count_syllables(verse, syllable_separator="|"):
    """
    Example input:
    |Nel |mez|zo |del |cam|min |di |no|stra |vi|ta

    Example output:
    11
    """
    syll = verse.split(syllable_separator)
    syll = list(filter(lambda x: x.strip() != "", syll))
    return len(syll)


def average_syllables(verses):
    """
    Takes a list of verses
    Returns the mean number of syllables among input verses
    """
    verse_count = len(verses)
    syll_counts = list(map(count_syllables, verses))
    syll_count = sum(syll_counts)

    return syll_count / verse_count


def correct_hendecasyllables_ratio(verses):
    """
    Takes a list of verses
    Returns the ratio of verses with a syllable count between 10 and 12
    """
    correct_verses = len(
        list(filter(lambda x: x <= 12 and x >= 10, map(count_syllables, verses)))
    )
    total_verses = len(verses)

    return correct_verses / total_verses


def is_vowel(c):
    return c in "aeiouAEIOUàèéìòù"


def find_termination(w):
    # TODO improve termination algorithm
    v_count = 0
    for i in range(len(w) - 1, -1, -1):
        if is_vowel(w[i]):
            v_count += 1
            if v_count == 2 and i < len(w) - 2:
                return w[i:]
            elif v_count == 3:
                return w[i:]


def are_in_rhyme(w1, w2):
    t1 = find_termination(w1)
    t2 = find_termination(w2)
    if t1 and t2 and t1 == t2 and len(t1) > 1 and len(t2) > 1:
        return True

    return False


def chained_rhymes_ratio(verses):
    """
    TODO only takes non syllabified verses
    TODO only takes verses without punctuation --> preprocess first!!

    Takes a list of verses

    Returns the ratio between correctly rhymed verses and total number of verses that can be verified (all but the last two verses)
    """
    total_verses = 0
    correct_rhymes = 0

    for i in range(len(verses) - 2):
        # The verse that
        if (i + 1) % 3 != 0:
            total_verses += 1
            correct_rhymes += are_in_rhyme(verses[i], verses[i + 2])

            if not are_in_rhyme(verses[i], verses[i + 2]):
                print(verses[i], verses[i + 2])

    return correct_rhymes / total_verses


def get_strophes(text, end_of_strophe="\n\n"):
    strophes = text.split(end_of_strophe)
    strophes = map(strip, strophes)
    strophes = filter(is_not_empty, strophes)
    return list(strophes)


def is_tercet(verses):
    verses = list(filter(lambda x: x.strip() != "", verses))
    return len(verses) == 3


def get_tercets(text, end_of_verse="\n"):
    strophes = get_strophes(text)

    # Divide each strophe according to the end of verse symbol, then count the verses
    strophes = map(lambda x: x.split(end_of_verse), strophes)
    strophes = filter(is_tercet, strophes)

    return list(strophes)


def tercet_to_strophe_ratio(verses):
    return len(get_tercets(verses)) / len(get_strophes(verses))


# TODO word correctness


# Adapted from https://github.com/AlessandroLiscio/DeepComedy
def ngrams_plagiarism(generated_text, original_text, n=4):
    # the tokenizer is used to remove non-alphanumeric symbols
    tokenizer = tfds.deprecated.text.Tokenizer()
    original_text = tokenizer.join(tokenizer.tokenize(original_text.lower()))
    generated_text_tokens = tokenizer.tokenize(generated_text.lower())

    total_ngrams = len(generated_text_tokens) - n + 1
    plagiarism_counter = 0

    for i in range(total_ngrams):
        ngram = tokenizer.join(generated_text_tokens[i : i + n])
        plagiarism_counter += 1 if ngram in original_text else 0
    return 1 - (plagiarism_counter / total_ngrams)