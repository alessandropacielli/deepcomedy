import nltk as nl
import numpy as np
import string
import collections
import os
import random
import re
import pickle

try:
    # Python 3
    from itertools import zip_longest
except ImportError:
    # Python 2
    from itertools import izip_longest as zip_longest


# GENERIC UTILS

def save_data(data, file):
    with open(file, 'wb') as output:
        pickle.dump(data, output, pickle.HIGHEST_PROTOCOL)

def load_data(file):
    with open(file, 'rb') as obj:
        return pickle.load(obj)

def print_and_write(file, s):
    print(s)
    file.write(s)



# NLP UTILS

_PAD = 0
_GO = 1
_EOW = 2
_UNK = 3
def to_chars(words, word_max_size):
    _PAD = 0
    _GO = 1
    _EOW = 2
    _UNK = 3
    chars = ['_PAD', '_GO', '_EOW', '_UNK', ' ', '!', '"', '#', '$', '%', '&', '\'', '(', ')',
             '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             ':', ';', '<', '=', '>', '?', '@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
             'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y',
             'Z', '[', '\\', ']', '^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
             'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y',
             'z', '{', '|', '}', 'ì', 'ò', 'ù', 'è', 'é', 'à']
    char_dict = {}
    for char in chars:
        char_dict[char] = len(char_dict)
    char_words = np.ndarray(shape=[len(words), word_max_size], dtype=np.int32)
    for i in range(len(words)):
        if words[i]=="<pad>":
            char_words[i][:] = _PAD
            continue
        char_words[i][0]=_GO
        for j in range(1,word_max_size):
            if j < len(words[i])+1:
                if words[i][j-1] in char_dict:
                    char_words[i][j] = char_dict[words[i][j-1]]
                else:
                    char_words[i][j] = _UNK
            elif j == len(words[i])+1:
                char_words[i][j] = _EOW
            else:
                char_words[i][j] = _PAD
        if char_words[i][word_max_size-1] != _PAD:
            char_words[i][word_max_size-1] = _EOW
    return char_words

'''
Read a file (filename) and return the textual content of the file in a vector of words
'''

def read_words(filename, max_len=None):
    try:
        nl.data.find('tokenizers/punkt')
    except LookupError:
        nl.download('punkt')
    with open(filename, "r") as f:
        st = f.read()
        st = st.translate(string.punctuation)
        data = nl.word_tokenize(st)
        del(st)
        if max_len:
            return data[:max_len]
        return data

def read_words_from_folder(data_path):
    try:
        nl.data.find('tokenizers/punkt')
    except LookupError:
        nl.download('punkt')
    list_files = [os.path.join(data_path, f)
                  for f in os.listdir(data_path)
                  if os.path.isfile(os.path.join(data_path, f))]
    words = []
    for filename in list_files:
        with open(filename, "r") as f:
            try:
                st = f.read()
            except UnicodeDecodeError:
                print("File "+filename+" decode error: SKIPPED")
                continue
            st = st.translate(string.punctuation)
            data = nl.word_tokenize(st)
            del(st)
            words.extend(data)
    return words


def build_dataset_of_tokens(tokens, vocabulary_size, special_tokens=[]):
    '''
    Given a list of tokens, it creates a dictionary mapping each token to a unique id.
    :param tokens: a list of strings.
     E.g. ["the", "cat", "is", ... ".", "the", "house" ,"is" ...].
     NB: Here you should put all your token instances of the corpus.
    :param vocabulary_size: The number of elements of your vocabulary. If there are more
    than 'vocabulary_size' elements on tokens, it considers only the 'vocabulary_size'
    most frequent ones.
    :param special_tokens: Optional. Useful to add special tokens in vocabulary. I
    f you don't have any, keep it empty.
    :return: data: the mapped tokens list;
     count: a dictionary containing the number of occurrences in 'tokens' for each
     element on your dictionary.
     dictionary: a python dictionary that associates a token with a unique integer identifier.
     reverse_dictionary: a python dictionary mapping a unique integer identifier to its token.
     E.g.
     dictionary:{"UNK": 0, "a": 1, "the": 2, ....}
     reverse_dictionary:{0:"UNK", 1:"a", 2:"the"}
    '''
    # counting occurrences of each token
    count = [['UNK', -1]]
    count.extend(collections.Counter(tokens).most_common(vocabulary_size - 1))  # takes only the most frequent ones
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    for token in special_tokens:
        dictionary[token[0]] = token[1]

    data = list()
    unk_count = 0
    for word in tokens:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def save_dictionary_tsv(filepath, dictionary, count):
    keys, values = zip(*count)
    with open(filepath, 'w') as f:
        f.write('Word\tFrequency\n')
        for k, v in dictionary.items():
            if k in keys:
                f.write(k + '\t' + str(values[keys.index(k)]) + '\n')
            else:
                f.write(k + '\t' + '0' + '\n')


def k_frequent(words_data, k):
    counter = collections.Counter(words_data)
    most = counter.most_common(k)
    res = [most[i][0] for i in range(len(most))]
    return res


def toWord(chars):
    str =''
    for c in chars:
        if(c<2):
            continue
        elif(c==2):
            break
        str = str+chr(c)
    return str


def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def get_dc_cantos(filename, encoding=None):
    # raw_data = read_words(filename=filename)
    cantos, words, raw = [], [], []
    with open(filename, "r", encoding=encoding) as f:
        for line in f:
            sentence = line.strip()
            sentence = str.replace(sentence, "\.", " \. ")
            sentence = str.replace(sentence, "[", '')
            sentence = str.replace(sentence, "]", '')
            sentence = str.replace(sentence, "-", '')
            sentence = str.replace(sentence, ";", " ; ")
            sentence = str.replace(sentence, ",", " , ")
            # sentence = str.replace(sentence, " \'", '')
            sentence = str.replace(sentence, "\'", ' \' ')
            if len(sentence) > 1:
                # sentence = sentence.translate(string.punctuation)
                tokenized_sentence = nl.word_tokenize(sentence)
                # tokenized_sentence = sentence.split()
                tokenized_sentence = [w.lower() for w in tokenized_sentence if len(w) > 0]
                tokenized_sentence = [w for w in tokenized_sentence if "," not in w]
                tokenized_sentence = [w for w in tokenized_sentence if "." not in w]
                tokenized_sentence = [w for w in tokenized_sentence if ":" not in w]
                tokenized_sentence = [w for w in tokenized_sentence if ";" not in w]
                tokenized_sentence = [w for w in tokenized_sentence if "«" not in w]
                tokenized_sentence = [w for w in tokenized_sentence if "»" not in w]
                # ts = []
                ts = tokenized_sentence
                # [ts.extend(re.split("(\')", e)) for e in tokenized_sentence]
                tokenized_sentence = [w for w in ts if len(w) > 0]

                if len(tokenized_sentence) == 2:
                    cantos.append([])
                    raw.append([])
                elif len(tokenized_sentence) > 2:
                    raw[-1].append(sentence)
                    cantos[-1].append(tokenized_sentence)
                    words.extend(tokenized_sentence)

    return cantos, words, raw


def get_cantica(filename, encoding=None):
    f_cantica = []
    count = 1
    with open(filename, "r", encoding=encoding) as f:
        for line in f:
            sentence = line.strip()
            tokenized_sentence = nl.word_tokenize(sentence)

            if len(tokenized_sentence) == 2:
                # setting feature cantica for each canto
                if count <= 34:
                    f_cantica.append([1, 0, 0])
                elif count > 34 and count <= 67:
                    f_cantica.append([0, 1, 0])
                else:
                    f_cantica.append([0, 0, 1])

                count += 1

    return f_cantica


def create_tercets(cantos):
    tercets = []
    for i,canto in enumerate(cantos):
        for v,verse in enumerate(canto):
            if v%3 == 0:
                tercets.append([])

            tercets[-1].append(verse)
        tercets = tercets[:-1]  # removes the last malformed tercets (only 2 verses)

    return tercets


def create_BAB_tercets(cantos):
    tercets = []
    for canto in cantos:
        for v,verse in enumerate(canto):
            if v%3 == 1:
                tercets.append([])
            if v > 0: tercets[-1].append(verse)

        tercets = tercets[:-1] # removes the last malformed tercets (only 2 verses)

    return tercets


def get_poetry(filename, encoding=None):
    # raw_data = read_words(filename=filename)

    raw_data, words = [], []
    # with open(filename, "r", encoding ='latin-1') as f:
    with open(filename, "r", encoding=encoding) as f:
        for line in f:
            sentence = line.strip()
            if len(sentence) > 1:
                sentence = sentence.translate(string.punctuation)
                tokenized_sentence = nl.word_tokenize(sentence)
                tokenized_sentence = [w.lower() for w in tokenized_sentence if len(w)>0]
                tokenized_sentence = [w for w in tokenized_sentence if "," not in w]
                tokenized_sentence = [w for w in tokenized_sentence if "." not in w]
                tokenized_sentence = [w for w in tokenized_sentence if ":" not in w]
                tokenized_sentence = [w for w in tokenized_sentence if ";" not in w]
                ts = []
                [ts.extend(re.split("(\')", e)) for e in tokenized_sentence]
                tokenized_sentence = [w for w in ts if len(w) > 0]

                if len(tokenized_sentence) > 2:
                    raw_data.append(tokenized_sentence)
                    words.extend(tokenized_sentence)

    return raw_data, words


def get_decameron(filename):
    raw_data, words = [], []
    with open(filename, "r", encoding='latin-1') as f:
        for line in f:
            raw_sentences = line.strip()
            if len(raw_sentences) > 1:
                sentences = raw_sentences.translate(string.punctuation)
                s_list = re.split("\.", sentences)
                for s in s_list:
                    tokenized_sentences = nl.word_tokenize(s)
                    tokenized_sentences = [w.lower() for w in tokenized_sentences if len(w)>0]
                    tokenized_sentences = [w for w in tokenized_sentences if "," not in w]
                    # tokenized_sentences = [w for w in tokenized_sentences if "." not in w]
                    tokenized_sentences = [w for w in tokenized_sentences if ":" not in w]
                    ts = []
                    [ts.extend(re.split("(\')", e)) for e in tokenized_sentences]
                    tokenized_sentences = [w for w in ts if len(w) > 0]

                    if len(tokenized_sentences) > 1 and len(tokenized_sentences) < 30:
                        raw_data.append(tokenized_sentences)
                        words.extend(tokenized_sentences)

    return raw_data, words


def build_dataset_from_dict(raw_data, dictionary, config, shuffle=True):
    '''
    Converts all the tokens in raw_data by mapping each token with its corresponding
    value in the dictionary. In case of token not in the dictionary, they are assigned to
    a specific id. Each sequence is padded up to the sentence_max_len setup in the config.

    :param raw_data: list of sequences, each sequences is a list of tokens (strings).
    :param dictionary: a python dictionary having as keys strings
    and int tokens as values.
    :param config: config object from class Config.
    :param shuffle: Optional. If True data are shuffled.
    :return: A list of sequences where each token in each sequence is an int id.
    '''
    dataset = []
    for sentence in raw_data:
        sentence_ids = [config._GO]
        sentence_ids.extend([dictionary[w] if w in dictionary else dictionary["UNK"] for w in sentence])
        sentence_ids.append(config._EOS)
        sentence_ids = pad_list(sentence_ids, config._PAD, config.sentence_max_len)

        dataset.append(sentence_ids)

    if shuffle:
        return random.sample(dataset, len(dataset))
    else:
        return dataset


def build_stanzas_dataset_from_dict(raw_data, dictionary, config, shuffle=True):
    dataset = []
    for stanza in raw_data:
        stanza_ids = [config._GO]
        for sentence in stanza:
            sentence_ids = [dictionary[w] if w in dictionary else dictionary["UNK"] for w in sentence]
            sentence_ids.append(config._EOS)
            stanza_ids.extend(sentence_ids)
        # stanza_ids.append(config._EOT)
        stanza_ids[-1] = config._EOT
        stanza_ids = pad_list(stanza_ids, config._PAD, config.sentence_max_len)

        dataset.append(stanza_ids)

    if shuffle:
        return random.sample(dataset, len(dataset))
    else:
        return dataset


def build_stanzas_dataset_from_subword_dict(raw_data, dictionary, config, shuffle=True):
    dataset = []
    for stanza in raw_data:
        stanza_ids = [pad_list([config._GO], config._PAD, config.word_max_len)]
        for sentence in stanza:
            for word in sentence:
                hyp_ids = [dictionary[hyp] if hyp in dictionary else dictionary["UNK"] for hyp in word]
                hyp_ids.append(config._EOW)
                hyp_ids = pad_list(hyp_ids, config._PAD, config.word_max_len)
                stanza_ids.extend([hyp_ids])

            stanza_ids.append(pad_list([config._EOS], config._PAD, config.word_max_len))
        # stanza_ids.append(config._EOT)
        stanza_ids[-1] = pad_list([config._EOT], config._PAD, config.word_max_len)
        stanza_ids = pad_list(stanza_ids, [config._PAD] * config.word_max_len, config.sentence_max_len)

        dataset.append(stanza_ids)

    if shuffle:
        return random.sample(dataset, len(dataset))
    else:
        return dataset


def build_stanzas_dataset_from_chars(raw_data, config):
    dataset = []
    for stanza in raw_data:
        stanza_ids = []
        for sentence in stanza:
            sentence_ids = [to_chars([w], config.word_max_len)[0] for w in sentence]
            sentence_ids.append(to_chars(["<EOS>"], config.word_max_len)[0])
            stanza_ids.extend(sentence_ids)
        stanza_ids.append(to_chars(["<EOT>"], config.word_max_len)[0])
        stanza_ids = pad_list(stanza_ids, to_chars(["<pad>"], config.word_max_len)[0], config.sentence_max_len)

        dataset.append(stanza_ids)
    return random.sample(dataset, len(dataset))


def pad_list(l, pad_token, max_l_size, keep_lasts=False, pad_right=True):
    """
    Adds a padding token to a list
    inputs:
    :param l: input list to pad.
    :param pad_token: value to add as padding.
    :param max_l_size: length of the new padded list to return,
    it truncates lists longer that 'max_l_size' without adding
    padding values.
    :param keep_lasts: If True, preserves the max_l_size last elements
    of a sequence (by keeping the same order).  E.g.:
    if keep_lasts is True and max_l_size=3 [1,2,3,4] becomes [2,3,4].


    :return: the list padded or truncated.
    """
    to_pad = []
    max_l = min(max_l_size, len(l))  # maximum len
    l_init = len(l) - max_l if len(l) > max_l and keep_lasts else 0  # initial position where to sample from the list
    l_end = len(l) if len(l) > max_l and keep_lasts else max_l
    for i in range(l_init, l_end):
        to_pad.append(l[i])

    # for j in range(len(l), max_l_size):
    #     to_pad.append(pad_token)
    pad_tokens = [pad_token] * (max_l_size-len(l))
    padded_l = to_pad + pad_tokens if pad_right else pad_tokens + to_pad

    return padded_l


def create_lm_target(x, config):
    return [e[1:] + [config._PAD] for e in x]


def create_hyp_lm_target(x, config):
    return [e[1:] + [[config._PAD] * config.word_max_len] for e in x]


def batches(x, y, batch_size=128):

    # Shuffle sentences
    sentences_ids = random.sample(range(len(x)), len(x))

    # Generator for batch
    batch_x, batch_y = [], []
    if batch_size is None:
        batch_size = len(x)
    for id in sentences_ids:
        batch_x.append(x[id])
        batch_y.append(y[id])
        if len(batch_x) % batch_size == 0:
            yield batch_x, batch_y
            batch_x, batch_y = [], []


def _batches(iterable, batch_size=128):

    # Shuffle sentences
    x = list(zip(*iterable))
    sentences_ids = random.sample(range(len(x)), len(x))

    # Generator for batch
    batch, batch_y = [], []
    if batch_size is None:
        batch_size = len(x)
    for id in sentences_ids:
        batch.append(x[id])
        if len(batch) % batch_size == 0:
            yield batch
            batch = []


def batches3(chars, x, y, batch_size=128):

    # Shuffle sentences
    sentences_ids = random.sample(range(len(x)), len(x))

    # Generator for batch
    batch_chars, batch_x, batch_y = [], [], []
    if batch_size is None:
        batch_size = len(x)
    for id in sentences_ids:
        batch_chars.append(chars[id])
        batch_x.append(x[id])
        batch_y.append(y[id])
        if len(batch_x) % batch_size == 0:
            yield batch_chars, batch_x, batch_y
            batch_chars, batch_x, batch_y = [], [], []


def are_in_rhyme(w1, w2):

    def find_termination(w):
        v_count = 0
        for i in range(len(w)-1, -1, -1):
            if is_vowel(w[i]):
                v_count += 1
                if v_count == 2 and i < len(w)-2:
                    # se è la seconda vocale che trovo, è una vocale e non è nel penultimo carattere
                    return w[i:]
                elif v_count == 3:
                    # se è la terza vocale che trovo
                    return w[i:]

    t1 = find_termination(w1)
    t2 = find_termination(w2)
    if t1 and t2 and t1 == t2 and len(t1) > 1 and len(t2) > 1:
        return True

    return False


def is_vowel(c):
    return c in 'aeiouAEIOUìèéùò'


def general_batches(iterables, full_size, batch_size=128):
    if batch_size == None:
        batch_size = full_size

    batch = []
    for i in range(len(iterables)):
        batch.append([])

    for id in range(full_size):
        for i,it in enumerate(iterables):
            batch[i].append(it[id])
        if len(batch[0]) % batch_size == 0:
            yield batch
            batch = []
            for i in range(len(iterables)):
                batch.append([])


def split_in_ngrams(x, n=3, pad_token='_'):
    '''
    Arguments:
    'x' a string of text, e.g. a word a sentence.
    'pad_token' the token to add to unfinished trigrams.

    Returns:
    a list containing 'x' split in trigrams.'''

    trigrams = []
    for i, ch in enumerate(x):
        if i % n == 0:
            trigrams.append('')

        trigrams[-1] += ch

    for p in range(3 - len(trigrams[-1])):
        trigrams[-1] += pad_token

    return trigrams


def get_ngrams(l, n=3, pad_token='_'):
    '''
    Arguments:
    'l' an input list of words.
    'pad_token' the token to add to unfinished trigrams.

    Returns:
    a list of the trigrams of 'l'.'''
    trigrams = []
    for w in l:
        t = split_in_ngrams(w, n, pad_token)
        trigrams.extend(t)

    return trigrams


def get_ngrams_from_tercets(tercets, n=3, pad_token='_'):
    '''
    Arguments:
    'tercets' a list of tercets.
    'pad_token' the token to add to unfinished trigrams.

    Returns:
    'tr_tercets' a list of tercets splitted in trigrams
    'trigrams' a list of all the trigrams'''
    tr_tercets, trigrams = [], []
    for tercet in tercets:
        tr_tercets.append([])
        for verse in tercet:
            # t = split_in_ngrams(verse, pad_token) # CHANGE IN get_ngrams IF tercets are not raw
            t = get_ngrams(verse, n, pad_token)
            tr_tercets[-1].append(t)
            trigrams.extend(t)

    return tr_tercets, trigrams


def get_ngrams_from_canzoniere(verses, n=3, pad_token='_'):
    '''
    Arguments:
    'verses' a list of verses.
    'pad_token' the token to add to unfinished trigrams.

    Returns:
    'tr_tercets' a list of tercets splitted in trigrams
    'trigrams' a list of all the trigrams'''
    tr_verses, trigrams = [], []
    for verse in verses:
        # t = split_in_ngrams(verse, pad_token) # CHANGE IN get_ngrams IF tercets are not raw
        t = get_ngrams(verse, n, pad_token)
        tr_verses.append(t)
        trigrams.extend(t)

    return tr_verses, trigrams


def get_sonnets(verses, N=14):
    sonnets, words = [], []
    for v, verse in enumerate(verses):
        if v % N == 0:
            sonnets.append([])

        tokenized_sentence = [w.lower().strip() for w in nl.word_tokenize(verse) if len(w) > 0]
        tokenized_sentence = [w for w in tokenized_sentence if "," not in w]
        tokenized_sentence = [w for w in tokenized_sentence if "." not in w]
        tokenized_sentence = [w for w in tokenized_sentence if ":" not in w]
        tokenized_sentence = [w for w in tokenized_sentence if ";" not in w]
        tokenized_sentence = [w for w in tokenized_sentence if "«" not in w]
        tokenized_sentence = [w for w in tokenized_sentence if "»" not in w]
        tokenized_sentence = [w for w in tokenized_sentence if "‘" not in w]
        tokenized_sentence = [w for w in tokenized_sentence if "‘‘" not in w]
        tokenized_sentence = [w for w in tokenized_sentence if "(" not in w]
        tokenized_sentence = [w for w in tokenized_sentence if ")" not in w]
        ts = []
        [ts.extend(re.split("(\')", e)) for e in tokenized_sentence]
        tokenized_sentence = [w for w in ts if len(w) > 0]
        sonnets[-1].append(tokenized_sentence)
        words.extend(tokenized_sentence)

    return sonnets, words


def get_quatrains(shake_sonnets):
    quatrains, words = [],[]
    for sonnet in shake_sonnets:
        quatrains.extend([sonnet[:4],sonnet[4:8], sonnet[:8:-2]])

    return quatrains


def load_paisa_data(filename, n_docs=15000):
    with open(filename, 'r', encoding="utf8") as f:
        it_data = []
        c = 0
        for line in f:
            if line[0] not in ['#', '<'] and len(line) > 150:
                line = re.sub('\d+', '0', line)
                line = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url', line)
                line = re.sub(',', ' , ', line)
                line = re.sub(';', ' ; ', line)
                line = re.sub(':', ' : ', line)
                line = re.sub('\(', ' ( ', line)
                line = re.sub('\)', ' ) ', line)
                line = re.sub("\'", " \' ", line)
                line = line.lower()
                it_data.append(line)
                if c >= n_docs:
                    break
                c += 1

        return it_data


def build_dataset_with_context(raw_data, dictionary, config, pad_len):
    dataset = []
    for sentence in raw_data:
        stanza_ids = [config._GO]
        sentence_ids = [dictionary[w] if w in dictionary else dictionary["UNK"] for w in sentence]
        sentence_ids.append(config._EOS)
        stanza_ids.extend(sentence_ids)
        stanza_ids = pad_list(stanza_ids, config._PAD, pad_len)

        dataset.append(stanza_ids)

    return dataset


def get_context_dataset(sequences):
    x_seq = []
    for seq in sequences:
        for i, c_verse in enumerate(seq):
            context = []
            [context.extend(v) for v in seq[:i]]
            x_seq.append((context, c_verse))

    return x_seq


def load_textual_corpus(filename, max_n_lines=-1):
    '''
    General function to load a textual file from corpus. A list of sentences
    is given as return, data is also cleaned up to remove urls and numbers. Sentences are
    split according to the dot '.' .
    :param filename: the name of the textual file to load.
    :param max_n_lines: Maximum number of lines to load from the file. Useful for huge corpora.
    Set to -1 (default), to get all the lines.
    :return: A list of sentences, where each sentence is a list of words.
    '''

    with open(filename, 'r', encoding="utf8", errors='ignore') as f:
        data = []
        c = 0
        for line in f:
            line = re.sub('\d+', '0', line)
            line = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'url', line)
            line = re.sub(',', ' , ', line)
            line = re.sub(';', ' ; ', line)
            line = re.sub(':', ' : ', line)
            line = re.sub('\(', ' ( ', line)
            line = re.sub('\)', ' ) ', line)
            line = re.sub("\'", " \' ", line)
            line = line.lower()
            data.extend([s.strip().split() for s in line.split(".") if len(s) > 2])

            if max_n_lines > 0 and c >= max_n_lines:
                break
            c += 1

        return data


def load_poetry_corpus(filename, scheme_n=3):
    with open(filename, 'r', encoding="utf8", errors='ignore') as f:
        data = []
        j = 0
        for line in f:
            if len(line) > 1:
                line = re.sub(',', ' , ', line)
                line = re.sub(';', ' ; ', line)
                line = re.sub(':', ' : ', line)
                line = re.sub('\(', ' ( ', line)
                line = re.sub('\)', ' ) ', line)
                line = re.sub("\'", " \' ", line)
                line = line.lower()
                tokenized_line = nl.word_tokenize(line)
                # tokenized_line = tokenized_line.split()
                tokenized_line = [w for w in tokenized_line if "," not in w]
                tokenized_line = [w for w in tokenized_line if "." not in w]
                tokenized_line = [w for w in tokenized_line if ":" not in w]
                tokenized_line = [w for w in tokenized_line if ";" not in w]
                tokenized_line = [w for w in tokenized_line if "!" not in w]
                tokenized_line = [w for w in tokenized_line if "?" not in w]
                tokenized_line = [w for w in tokenized_line if "«" not in w]
                tokenized_line = [w for w in tokenized_line if "»" not in w]

                if j % scheme_n == 0:
                    data.append([])
                data[-1].append(tokenized_line)

                j += 1

        return data


def load_dantes_poetry(filenames, stanza_size=3):
    '''

    :param filenames: A list of filenames containing raw textual Dante's data.
    :param stanza_size: The number of verses to group together.
    :return: a list of stanzas, each stanza is a list of verses
    '''

    poetries = []
    for filename in filenames:
        poetries.extend(load_poetry_corpus(filename, stanza_size))

    return poetries


def load_dantes_prose(filenames, max_n_lines=-1):
    '''
    Function to retrieve Dante's prose.
    :param filenames: A list of filenames containing raw textual Dante's data.
    :param max_n_lines: Maximum number of lines to load from the files.
    Default value -1, indicates no limit.
    :return: A list of sentences.
    '''

    prose = []
    for filename in filenames:
        sentences = load_textual_corpus(filename, max_n_lines)
        prose.extend(sentences)

    return prose

