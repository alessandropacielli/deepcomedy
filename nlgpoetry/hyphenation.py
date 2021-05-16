import os
from .utils import get_dc_cantos, create_tercets, print_and_write, get_cantica
# from utils import get_dc_cantos, create_tercets, print_and_write, get_cantica
# from Utils.utils import get_dc_cantos, create_tercets, print_and_write
# from utils import get_dc_cantos, create_tercets, print_and_write


# SILLABATOR

def is_vowel(c):
    return c in 'aeiouAEIOUàìíèéùúüòï'


def get_vowels(w):
    return [(c, i) for i, c in enumerate(w) if is_vowel(c)]


def unsplittable_cons():
    u_cons = []
    for c1 in ('b', 'c', 'd', 'f', 'g', 'p', 't', 'v'):
        for c2 in ('l', 'r'):
            u_cons.append(c1 + c2)

    others = ['gn', 'gh', 'ch']
    u_cons.extend(others)
    return u_cons


def are_cons_to_split(c1, c2):
    to_split = ('cq', 'cn', 'lm', 'rc', 'bd', 'mb', 'mn', 'ld', 'ng', 'nd', 'tm', 'nv', 'nc', 'ft', 'nf', 'gm', 'fm', 'rv', 'fp')
    return (c1 + c2) in to_split or (not is_vowel(c1) and (c1 == c2)) or ((c1 + c2) not in unsplittable_cons()) and (
        (not is_vowel(c1)) and (not is_vowel(c2)) and c1 != 's')


def is_diphthong(c1, c2):
    return (c1 + c2) in ('ia', 'ie', 'io', 'iu', 'ua', 'ue', 'uo', 'ui', 'ai', 'ei', 'oi', 'ui', 'au', 'eu', 'ïe', 'iú', 'iù')


def is_triphthong(c1, c2, c3):
    return (c1 + c2 + c3) in ('iai', 'iei', 'uoi', 'uai', 'uei', 'iuo')


def is_toned_vowel(c):
    return c in 'àìèéùòï'


def get_next_vowel_pos(word, start_pos=0):
    c = word[start_pos]
    count = start_pos
    while not is_vowel(c) or count == len(word):
        count += 1
        c = word[count]

    return count + 1


def has_vowels(sy):
    for c in sy:
        if is_vowel(c):
            return True
    return False


def hyphenation(word):
    """
    Split word in syllables
    :param word: input string
    :return: a list containing syllables of the word
    """
    if not word or word == '':
        return []
    # elif len(word) == 3 and (is_vowel(word[1]) and is_vowel(word[2]) and not is_toned_vowel(word[2]) and (
    #     not is_diphthong(word[1], word[2]) or (word[1] == 'i'))):
    elif len(word) == 3 and (is_vowel(word[1]) and is_vowel(word[2]) and not is_toned_vowel(word[2]) and (
        not is_diphthong(word[1], word[2]))):
        return [word[:2]] + [word[2]]
    elif len(word) == 3 and is_vowel(word[0]) and not is_vowel(word[1]) and is_vowel(word[2]):
        return [word[:2]] + [word[2]]
    elif len(word) == 3:
        return [word]

    syllables = []
    is_done = False
    count = 0
    while not is_done and count <= len(word) - 1:
        syllables.append('')
        c = word[count]
        while not is_vowel(c) and count < len(word) - 1:
            syllables[-1] = syllables[-1] + c
            count += 1
            c = word[count]

        syllables[-1] = syllables[-1] + word[count]

        if count == len(word) - 1:
            is_done = True
        else:
            count += 1

            if count < len(word) and not is_vowel(word[count]):
                if count == len(word) - 1:
                    syllables[-1] += word[count]
                    count += 1
                elif count + 1 < len(word) and are_cons_to_split(word[count], word[count + 1]):
                    syllables[-1] += word[count]
                    count += 1
                elif count + 2 < len(word) and not is_vowel(word[count + 1]) and not is_vowel(word[count + 2]) and word[
                    count] != 's':
                    syllables[-1] += word[count]
                    count += 1
            elif count < len(word):
                if count + 1 < len(word) and is_triphthong(word[count - 1], word[count], word[count + 1]):
                    syllables[-1] += word[count] + word[count + 1]
                    count += 2
                elif is_diphthong(word[count - 1], word[count]):
                    syllables[-1] += word[count]
                    count += 1

                if count + 1 < len(word) and are_cons_to_split(word[count], word[count + 1]):
                    syllables[-1] += word[count]
                    count += 1

            else:
                is_done = True

    if not has_vowels(syllables[-1]) and len(syllables) > 1:
        syllables[-2] = syllables[-2] + syllables[-1]
        syllables = syllables[:-1]

    return syllables


def seq_hyphentation(words):
    """
    Converts words in a list of strings into lists of syllables
    :param words: a list of words (strings)
    :return: a list of lists containing word syllables
    """
    return [hyphenation(w) for w in words]


def get_seq_hyphen_len(words):
    return sum([len(hyphenation(w)) for w in words])


def get_hyp_lm_tercets(tercets):
    new_tercets = []
    for tercet in tercets:
        new_tercets.append([])
        for verse in tercet:
            new_tercets[-1].append([])
            for hyp_w in verse:
                new_tercets[-1][-1].extend(hyp_w)
                new_tercets[-1][-1].append('<SEP>')
            new_tercets[-1][-1] = new_tercets[-1][-1][:-1]

    return new_tercets


def get_dc_hyphenation(canti):
    hyp_canti, hyp_tokens = [], []
    for canto in canti:
        hyp_canti.append([])
        for verso in canto:
            syllables = seq_hyphentation(verso)
            hyp_canti[-1].append(syllables)
            for syllable in syllables:
                hyp_tokens.extend(syllable)

    return hyp_canti, hyp_tokens


def hyp2word(hyphen, hyp_rev_vocabulary, special_tokens):
    word = ''
    for hyp in hyphen:
        if hyp not in special_tokens and hyp in hyp_rev_vocabulary:
            word += hyp_rev_vocabulary[hyp]
        elif hyp not in special_tokens:
            word += '<UNK>'

    return word


def get_hyps(batch, hyp_rev_vocabulary, special_tokens):
    hyps = []
    for seq in batch:
        hyps.append('')
        for hyphen in seq:
            hyps[-1] += hyp2word(hyphen, hyp_rev_vocabulary, special_tokens) + ' '

    return hyps


def print_hyps(batch, hyp_rev_vocabulary, special_tokens):
    for seq in batch:
        to_print = ''
        for hyphen in seq:
            to_print.join(hyp2word(hyphen, hyp_rev_vocabulary, special_tokens) + ' ')
        print(to_print)


def print_paired_hyps(file, batch_y, batch_z, hyp_rev_vocabulary, special_tokens):
    hyps_y = get_hyps(batch_y, hyp_rev_vocabulary, special_tokens)
    hyps_z = get_hyps(batch_z, hyp_rev_vocabulary, special_tokens)

    for i in range(len(hyps_y)):
        print_and_write(file, 'Ground Truth: ' + hyps_y[i])
        print_and_write(file, 'Prediction: ' + hyps_z[i])


def print_paired_output(file, batch_y, batch_z, rev_vocabulary, special_tokens, end_of_tokens=None):

    def output2string(batch, rev_vocabulary, special_tokens, end_of_tokens):
        output_strings = []
        for seq in batch:
            to_print = ''
            for token in seq:
                if token in special_tokens:
                    to_print += ' '
                elif end_of_tokens and token in end_of_tokens:
                    to_print += '\n'
                elif token in rev_vocabulary:
                    to_print += rev_vocabulary[token]
                else:
                    to_print += '<UNK>'
            output_strings.append(to_print)

        return output_strings

    hyps_y = output2string(batch_y, rev_vocabulary, special_tokens, end_of_tokens)
    hyps_z = output2string(batch_z, rev_vocabulary, special_tokens, end_of_tokens)

    for i in range(len(hyps_y)):
        print_and_write(file, "\n================================================")
        print_and_write(file, 'Ground Truth: ' + hyps_y[i] + "\n")
        print_and_write(file, 'Prediction: ' + hyps_z[i] + "\n")
        print_and_write(file, "================================================\n")


def hyps2words(ids, sep, pad=-1, with_sep=False, omit_pad=True):
    """
    Splits the list of ids according to a separator.

    :param ids: a list of hyphen' ids
    :param sep: the separator token (INT value)
    :param pad (optional): id of the pad token (INT value)
    :param with_sep (optional): separators are omitted if True,
    otherwise they are kept
    :param omit_pad (optional): true or false to decide whether
    to omit pad token or not
    :return: a list of elements, where each element
    is a list of tokens composing a word
    """

    words = [[]]
    for id in ids:
        if id == sep:
            if with_sep:
                words.append([sep])
            words.append([])
        elif id != pad or (id == pad and not omit_pad):
            words[-1].append(id)

    return words


def hyps2word(hyps):
    """
    Converts a list of hyphens to a string.
    :param hyps: a list of strings (hyphens)
    :return: string of concatenated hyphens
    """

    return ''.join(hyps)


def id2hyp(id, rev_dictionary):
    """
    Converts an id to its respective hyphen in rev_dictionary.
    :param id: an integer
    :param rev_dictionary: a Python dictionary
    with integer as keys and strings as values.
    :return: a string
    """
    return rev_dictionary[id] if id in rev_dictionary else '<UNK>'


def hyp2id(hyp, dictionary):
    """
        Converts an hyphen to its respective id in dictionary.
        :param hyp: a string
        :param dictionary: a Python dictionary
        with string as keys and integers as values.
        :return: an integer
        """
    return dictionary[hyp] if hyp in dictionary else 0


def ids2hyps(ids, rev_dictionary):
    """
    Maps a list of ids into a list of hyphens.
    :param ids: a list of ints
    :param rev_dictionary:  Python dictionary
    with string as keys and integers as values.
    :return: a list of strings (hyphens)
    """
    return [id2hyp(id, rev_dictionary) for id in ids]


def is_word(hyps, word_dictionary):
    return hyps2word(hyps) in word_dictionary


def hyps2verses(ids, eos, eot):
    """
    Split the list of hypens in different lists, separated
    by the sep token.
    :param ids: a list of hyphen' ids
    :param eos: the separator token (INT) (id corresponding to <EOS>)
    :return: a list of verses, each verse is a list of syllables
    """

    verses = [[]]
    for id in ids:
        if id == eot:
            break
        elif id == eos:
            verses.append([])
        else:
            verses[-1].append(id)

    if len(verses[-1]) < 1:
        verses = verses[:-1]

    return verses


def hyphenize_list(l):
    """
    Given a corpus, the function tokenizes it by dividing words into syllables
    adding also a separator token between words.
    :param l: a list of sequences, each sequence is a list of words (strings).
    :return: a list of sequences, but each sequence is a list of syllables.
    """

    sentences = [seq_hyphentation(s) for s in l]
    sep_token = "<SEP>"
    hyphenated_sentences = []
    for s in sentences:
        hyphenated_sentences.append([])
        for w in s:
            hyphenated_sentences[-1].extend(w)
            hyphenated_sentences[-1].append(sep_token)
        hyphenated_sentences[-1] = hyphenated_sentences[-1][:-1]  # removes last sep_token

    return hyphenated_sentences


if __name__ == '__main__':

    filename = os.path.join(os.getcwd(), "..", "datasets", "la_divina_commedia.txt")
    canti, _, raw = get_dc_cantos(filename=filename, encoding='latin-1')
    cantica = get_cantica(filename=filename, encoding='latin-1')
    canti, tokens = get_dc_hyphenation(canti)

    tercets, f_cantica = create_tercets(list(zip(canti, cantica)))
    acc = 0
    tot = 0
    for t in tercets:
        for v in t:
            clean_v = [[s for s in w if s not in "!?,;-'\'\""] for w in v]
            clean_v = [w for w in clean_v if len(w) > 0]
            with_sinalefe_c, no_sinalefe_c = 0, 0
            v_syllables = []
            prev_sy = ['!']
            for w in clean_v:
                v_syllables.append(w)
                # print(syllables)
                if is_vowel(prev_sy[-1][-1]) and is_vowel(w[0][0]) and not ((prev_sy[-1][-1] + w[0][0]) in ['ài', 'éa', 'ìa', 'ùo', 'òi']):
                    with_sinalefe_c += len(w) - 1
                else:
                    with_sinalefe_c += len(w)
                no_sinalefe_c += len(w)

                prev_sy = w


            # print('Number of syllables:', with_sinalefe_c)
            if with_sinalefe_c == 11 or no_sinalefe_c == 11:
                acc += 1
            else:
                print("Seems to be an error")
                print(with_sinalefe_c)
                print(no_sinalefe_c)
                # print(v)
                print(v_syllables)
            tot += 1

    print('Accuracy', float(acc)/tot)