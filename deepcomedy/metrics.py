from .utils import is_not_empty, strip
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import numpy as np
from strsimpy.normalized_levenshtein import NormalizedLevenshtein


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


# Adapted from https://gitlab.com/zugo91/nlgpoetry
def is_vowel(c):
    return c in "aeiouAEIOUàèéìòù"


# Adapted from https://gitlab.com/zugo91/nlgpoetry
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


# Adapted from https://gitlab.com/zugo91/nlgpoetry
def are_in_rhyme(w1, w2):
    t1 = find_termination(w1)
    t2 = find_termination(w2)
    if t1 and t2 and t1 == t2 and len(t1) > 1 and len(t2) > 1:
        return True

    return False


def chained_rhymes_ratio(verses, verbose=False):
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
            
            # If verbose print incorrect verses
            if verbose and not are_in_rhyme(verses[i], verses[i + 2]):
                print("{}\t{}".format(verses[i], verses[i + 2]))

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
    return plagiarism_counter / total_ngrams


def correct_words_ratio(gen_words, real_words, return_errors=False):
    exact_matches = np.array([word in real_words for word in gen_words], dtype=bool)
    n_exact_matches = sum(exact_matches)

    ratio = n_exact_matches / len(gen_words)

    if return_errors:

        return ratio, np.array(list(gen_words))[~exact_matches]

    return ratio


# Adapted from https://github.com/AlessandroLiscio/DeepComedy
def find_similar_words(
    word: str, vocabulary: set, verbose=False, return_best_distance=False
):
    """
    Given a word, find the most similar words in the vocabulary. \n
    The similarity between words is computed by the 'edit (Levenshtein) distance'. \n
    If more than one word from the vocabulary has a same distance from the requested word, \n
    then a list containing them is returned. \n
    If #return_best_distance is True, then returns a tuple containing the best words and the best distance.
    """
    if verbose:
        print(f"looking for words similar to '{word}'")
    # try normal search
    if word in vocabulary:
        if verbose:
            print(f"\t{(word, 0)} <-- match")
        return ([word], 0) if return_best_distance else [word]

    else:
        most_similar = []
        best_distance = len(
            word
        )  # the distance between a word and an empty word is equal to the lenght of the word itself

        for real_word in vocabulary:
            dist = word_distance(word, real_word)
            if dist <= best_distance:
                if verbose:
                    print(f"\t {real_word} ({dist})")
                if dist < best_distance:
                    best_distance = dist
                    most_similar = []
                most_similar.append(real_word)
        return (most_similar, best_distance) if return_best_distance else most_similar


# Adapted from https://github.com/AlessandroLiscio/DeepComedy
def word_distance(a, b):
    from nltk.metrics import edit_distance

    d = edit_distance(a, b)
    return d


# Adapted from https://github.com/AlessandroLiscio/DeepComedy
def incorrectness(
    words: set,
    real_words: set,
    verbose=False,
    return_match_ratio=False,
    plot_frequencies=False,
):
    """
    Measures the amount of incorrect words, with respect to the given set of real words.
    If all the passed words exists in the real words set, then the returned value is 0, otherwise return a positive real number.
    The score is computed as a weighted average of the frequencies of the distances of the words from the real words.
    A set of words each of which has the nearest word (in Levenshtein distance) at 1 or 2 is way better of another one
    whose nearest word is 10 points far from the most similar real word.
    """
    if verbose:
        print(
            "{}\n{:3}\t{:10}\t{:5}\t{}\n{}\n".format(
                "=" * 40, "%", "WORD", "DIST", "SIMILAR TO", "=" * 40
            )
        )
    n_real_words = len(words)
    # compute frequencies
    distances = []
    for i, my_word in enumerate(words):
        most_similar, distance = find_similar_words(
            word=my_word, vocabulary=real_words, return_best_distance=True
        )
        distances.append(distance)

    # compute frequencies
    frequencies = dict(
        zip(np.unique(distances), [distances.count(d) for d in np.unique(distances)])
    )

    # add the zero-frequency if not present (in order to compute the correct words percentage)
    if 0 not in frequencies.keys():
        frequencies[0] = 0

    if verbose:
        print("\n{}\n frequencies: {}".format("-" * 40, dict(frequencies)))

    # computing correctness
    incorrectness = round(
        np.average(
            np.unique(distances),
            weights=[distances.count(d) for d in np.unique(distances)],
        ),
        2,
    )

    # percentage of incorrect words
    ratio = 1 - round(frequencies[0] / len(words), 2)

    # print final results
    if verbose:
        print(
            " match ratio:  {} %\t({} / {}) \n{}".format(
                ratio, frequencies[0], len(words), "=" * 40
            )
        )
        if distance != 0:
            print(
                "{:>3}\t{:15}\t{:<5}\t{}".format(
                    round(i / n_real_words * 100, 1), my_word, distance, most_similar
                )
            )
    # plot
    if plot_frequencies:
        from matplotlib import pyplot as pyplot

        plt.bar(list(frequencies), [frequencies[key] for key in list(frequencies)])
        plt.show()

    return (incorrectness, ratio) if return_match_ratio else incorrectness


def validate_syllabification(prod, target):
    """
    Evaluates the correctness of produced syllabification with a correct reference.

    prod: list[string] produced syllabification as a list of verses
    target: list[string] correct syllabification as a list of verses

    returns for each verse, whether or not the produced verse is correctly syllabified (exact match)
        and the edit distance between the produced string and the target

    """

    levenshtein = NormalizedLevenshtein()
    return [(x == y, levenshtein.similarity(x, y)) for x, y in zip(prod, target)]