from __future__ import division, unicode_literals

"""
Utility functions.
"""

import json
import os
import numpy as np
import nltk
from collections import defaultdict, Counter
from nltk.tokenize.regexp import RegexpTokenizer
from SNLIModel import SNLIModel

tokenizer = nltk.tokenize.TreebankWordTokenizer()

UNKNOWN = '**UNK**'
PADDING = '**PAD**'
START = '**START**'  # it's called "START" but actually serves as a null alignment


class RTEDataset(object):
    """
    Class for better organizing a data set. It provides a separation between
       first and second sentences and also their sizes.
    """

    def __init__(self, sentences1, sentences2, sizes1, sizes2, labels):
        """
        :param sentences1: A 2D numpy array with sentences (the first in each
            pair) composed of token indices
        :param sentences2: Same as above for the second sentence in each pair
        :param sizes1: A 1D numpy array with the size of each sentence in the
            first group. Sentences should be filled with the PADDING token after
            that point
        :param sizes2: Same as above
        :param labels: 1D numpy array with labels as integers
        """
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.sizes1 = sizes1
        self.sizes2 = sizes2
        self.labels = labels
        self.num_items = len(sentences1)

    def shuffle_data(self):
        """
        Shuffle all data using the same random sequence.
        :return:
        """
        shuffle_arrays(self.sentences1, self.sentences2,
                       self.sizes1, self.sizes2, self.labels)

    def get_batch(self, from_, to):
        """
        Return an RTEDataset object with the subset of the data contained in
        the given interval. Note that the actual number of items may be less
        than (`to` - `from_`) if there are not enough of them.

        :param from_: which position to start from
        :param to: which position to end
        :return: an RTEDataset object
        """
        if from_ == 0 and to >= self.num_items:
            return self

        subset = RTEDataset(self.sentences1[from_:to],
                            self.sentences2[from_:to],
                            self.sizes1[from_:to],
                            self.sizes2[from_:to],
                            self.labels[from_:to])
        return subset


def tokenize_english(text):
    """
    Tokenize a piece of text using the Treebank tokenizer

    :return: a list of strings
    """
    return tokenizer.tokenize(text)


def tokenize_portuguese(text):
    """
    Tokenize the given sentence in Portuguese. The tokenization is done in
    conformity  with Universal Treebanks (at least it attempts so).

    :param text: text to be tokenized, as a string
    """
    tokenizer_regexp = r'''(?ux)
    # the order of the patterns is important!!
    # more structured patterns come first
    [a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+|    # emails
    (?:[\#@]\w+)|                     # Hashtags and twitter user names
    (?:[^\W\d_]\.)+|                  # one letter abbreviations, e.g. E.U.A.
    (?:[DSds][Rr][Aa]?)\.|            # common abbreviations such as dr., sr., sra., dra.
    \b\d+(?:[-:.,]\w+)*(?:[.,]\d+)?\b|
        # numbers in format 999.999.999,999, or hyphens to alphanumerics
    \.{3,}|                           # ellipsis or sequences of dots
    (?:\w+(?:\.\w+|-\d+)*)|           # words with dots and numbers, possibly followed by hyphen number
    -+|                               # any sequence of dashes
    \S                                # any non-space character
    '''
    tokenizer = RegexpTokenizer(tokenizer_regexp)

    return tokenizer.tokenize(text)


def tokenize_corpus(pairs):
    """
    Tokenize all pairs.

    :param pairs: a list of tuples (sent1, sent2, relation)
    :return: a list of tuples as in pairs, except both sentences are now lists
        of tokens
    """
    tokenized_pairs = []
    for sent1, sent2, label in pairs:
        tokens1 = tokenize_english(sent1)
        tokens2 = tokenize_english(sent2)
        tokenized_pairs.append((tokens1, tokens2, label))

    return tokenized_pairs


def count_corpus_tokens(pairs):
    """
    Examine all pairs ans extracts all tokens from both text and hypothesis.

    :param pairs: a list of tuples (sent1, sent2, relation) with tokenized
        sentences
    :return: a Counter of lowercase tokens
    """
    c = Counter()
    for sent1, sent2, _ in pairs:
        c.update(t.lower() for t in sent1)
        c.update(t.lower() for t in sent2)

    return c


def shuffle_arrays(*arrays):
    """
    Shuffle all given arrays with the same RNG state.

    All shuffling is in-place, i.e., this function returns None.
    """
    rng_state = np.random.get_state()
    for array in arrays:
        np.random.shuffle(array)
        np.random.set_state(rng_state)


def create_label_dict(pairs):
    """
    Return a dictionary mapping the labels found in `pairs` to numbers
    :param pairs: a list of tuples (_, _, label), with label as a string
    :return: a dict
    """
    labels = set(pair[2] for pair in pairs)
    mapping = zip(labels, range(len(labels)))
    return dict(mapping)


def convert_labels(pairs, label_map):
    """
    Return a numpy array representing the labels in `pairs`

    :param pairs: a list of tuples (_, _, label), with label as a string
    :param label_map: dictionary mapping label strings to numbers
    :return: a numpy array
    """
    return np.array([label_map[pair[2]] for pair in pairs], dtype=np.int32)


def create_dataset(pairs, word_dict, label_dict=None,
                   max_len1=None, max_len2=None):
    """
    Generate and return a RTEDataset object for storing the data in numpy format.

    :param pairs: list of tokenized tuples (sent1, sent2, label)
    :param word_dict: a dictionary mapping words to indices
    :param label_dict: a dictionary mapping labels to numbers. If None,
        labels are ignored.
    :param max_len1: the maximum length that arrays for sentence 1
        should have (i.e., time steps for an LSTM). If None, it
        is computed from the data.
    :param max_len2: same as max_len1 for sentence 2
    :return: RTEDataset
    """
    tokens1 = [pair[0] for pair in pairs]
    tokens2 = [pair[1] for pair in pairs]
    sentences1, sizes1 = _convert_pairs_to_indices(tokens1, word_dict,
                                                   max_len1)
    sentences2, sizes2 = _convert_pairs_to_indices(tokens2, word_dict,
                                                   max_len2)
    if label_dict is not None:
        labels = convert_labels(pairs, label_dict)
    else:
        labels = None

    return RTEDataset(sentences1, sentences2, sizes1, sizes2, labels)


def _convert_pairs_to_indices(sentences, word_dict, max_len=None,
                              use_null=True):
    """
    Convert all pairs to their indices in the vector space.

    The maximum length of the arrays will be 1 more than the actual
    maximum of tokens when using the NULL symbol.

    :param sentences: list of lists of tokens
    :param word_dict: mapping of tokens to indices in the embeddings
    :param max_len: maximum allowed sentence length. If None, the
        longest sentence will be the maximum
    :param use_null: prepend a null symbol at the beginning of each
        sentence
    :return: a tuple with a 2-d numpy array for the sentences and
        a 1-d array with their sizes
    """
    sizes = np.array([len(sent) for sent in sentences])
    if use_null:
        sizes += 1
        if max_len is not None:
            max_len += 1

    if max_len is None:
        max_len = sizes.max()
    # max_len is the length of the longest sentence if None
    shape = (len(sentences), max_len)
    # make matrix of number_of_sentences X maximum_length_of_sentence where each value is index of PAD word.
    array = np.full(shape, word_dict[PADDING], dtype=np.int32)

    # Fill the array with the actual indices up to the length of each sentence.
    for i, sent in enumerate(sentences):
        indices = [word_dict[token] for token in sent]

        if use_null:
            indices = [word_dict[START]] + indices

        array[i, :len(indices)] = indices

    return array, sizes


def load_parameters(dirname):
    """
    Load a dictionary containing the parameters used to train an instance
    of the autoencoder.

    :param dirname: the path to the directory with the model files.
    :return: a Python dictionary
    """
    filename = os.path.join(dirname, 'model-params.json')
    with open(filename, 'rb') as f:
        data = json.load(f)

    return data


def get_sentence_sizes(pairs):
    """
    Count the sizes of all sentences in the pairs
    :param pairs: a list of tuples (sent1, sent2, _). They must be
        tokenized
    :return: a tuple (sizes1, sizes2), as two numpy arrays
    """
    sizes1 = np.array([len(pair[0]) for pair in pairs])
    sizes2 = np.array([len(pair[1]) for pair in pairs])
    return (sizes1, sizes2)


def get_max_sentence_sizes(pairs1, pairs2):
    """
    Find the maximum length among the first and second sentences in both
    pairs1 and pairs2. The two lists of pairs could be the train and validation
    sets

    :return: a tuple (max_len_sentence1, max_len_sentence2)
    """
    train_sizes1, train_sizes2 = get_sentence_sizes(pairs1)
    valid_sizes1, valid_sizes2 = get_sentence_sizes(pairs2)
    train_max1 = max(train_sizes1)
    valid_max1 = max(valid_sizes1)
    max_size1 = max(train_max1, valid_max1)
    train_max2 = max(train_sizes2)
    valid_max2 = max(valid_sizes2)
    max_size2 = max(train_max2, valid_max2)

    return max_size1, max_size2


def normalize_embeddings(embeddings):
    """
    Normalize the embeddings to have norm 1.
    :param embeddings: 2-d numpy array
    :return: normalized embeddings
    """
    # normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1).reshape((-1, 1))
    return embeddings / norms


def write_word_dict(word_dict, dirname):
    """
    Write the word dictionary to a file.

    It is understood that unknown words are mapped to 0.
    """
    words = [word for word in word_dict.keys() if word_dict[word] != 0]
    sorted_words = sorted(words, key=lambda x: word_dict[x])
    text = '\n'.join(sorted_words)
    path = os.path.join(dirname, 'word-dict.txt')
    with open(path, 'wb') as f:
        f.write(text.encode('utf-8'))


def read_word_dict(dirname):
    """
    Read a file with a list of words and generate a defaultdict from it.
    """
    filename = os.path.join(dirname, 'word-dict.txt')
    with open(filename, 'rb') as f:
        text = f.read().decode('utf-8')

    words = text.splitlines()
    index_range = range(1, len(words) + 1)
    return defaultdict(int, zip(words, index_range))


def write_extra_embeddings(embeddings, dirname):
    """
    Write the extra embeddings (for unknown, padding and null)
    to a numpy file. They are assumed to be the first three in
    the embeddings model.
    """
    path = os.path.join(dirname, 'extra-embeddings.npy')
    np.save(path, embeddings[:3])


def _generate_random_vector(size):
    """
    Generate a random vector from a uniform distribution between
    -0.1 and 0.1.
    """
    return np.random.uniform(-0.1, 0.1, size)


def load_embeddings(embeddings_path):
    """
    Load and return an embedding model in either text format or
    numpy binary format. The text format is used if vocabulary_path
    is None (because the vocabulary is in the same file as the
    embeddings).

    :param embeddings_path: path to embeddings file
    :param vocabulary_path: path to text file with vocabulary,
        if needed
    :param generate: whether to generate random embeddings for
        unknown, padding and null
    :param load_extra_from: path to directory with embeddings
        file with vectors for unknown, padding and null
    :param normalize: whether to normalize embeddings
    :return: a tuple (defaultdict, array)
    """

    print('Loading embeddings')
    wordlist, embeddings = load_text_embeddings(embeddings_path)

    # mapping every word to tuple of (word, index) where the indices starts from 3 up to len(wordlist) + 3
    # saving indices 0-2 for special cases.
    mapping = zip(wordlist, range(3, len(wordlist) + 3))

    # always map OOV words to 0
    wd = defaultdict(int, mapping)
    wd[UNKNOWN] = 0
    wd[PADDING] = 1
    wd[START] = 2
    # generating 3 random vectors for unknown, padding, start
    vector_size = embeddings.shape[1]
    extra = [_generate_random_vector(vector_size),
             _generate_random_vector(vector_size),
             _generate_random_vector(vector_size)]

    embeddings = np.append(extra, embeddings, 0)
    embeddings = normalize_embeddings(embeddings)
    # wd is a dictionary that maps from word to index
    return wd, embeddings


def load_binary_embeddings(embeddings_path, vocabulary_path):
    """
    Load any embedding model in numpy format, and a corresponding
    vocabulary with one word per line.

    :param embeddings_path: path to embeddings file
    :param vocabulary_path: path to text file with words
    :return: a tuple (wordlist, array)
    """
    vectors = np.loadtxt(embeddings_path)

    with open(vocabulary_path, 'rb') as f:
        text = f.read().decode('utf-8')
    words = text.splitlines()

    return words, vectors


def load_text_embeddings(path):
    """
    Load any embedding model written as text, in the format:
    word[space or tab][values separated by space or tab]

    :param path: path to embeddings file
    :return: a tuple (wordlist, array)
    """
    words = []

    # start from index 1 and reserve 0 for unknown
    vectors = []
    with open(path, 'r') as f:
        for line in f:
            line = line.decode('utf-8')
            line = line.strip()
            if line == '':
                continue

            fields = line.split(' ')
            word = fields[0]
            words.append(word)
            vector = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            vectors.append(vector)  # vectors is list of numpy arrays
    # make embeddings as numpy array (numpy array of numpy arrays)
    embeddings = np.array(vectors, dtype=np.float32)

    return words, embeddings


def write_params(dirname, lowercase, language=None, model='mlp'):
    """
    Write system parameters (not related to the networks) to a file.
    """
    path = os.path.join(dirname, 'system-params.json')
    data = {'lowercase': lowercase,
            'model': model}
    if language:
        data['language'] = language
    with open(path, 'wb') as f:
        json.dump(data, f)


def write_label_dict(label_dict, dirname):
    """
    Save the label dictionary to the save directory.
    """
    path = os.path.join(dirname, 'label-map.json')
    with open(path, 'wb') as f:
        json.dump(label_dict, f)


def load_label_dict(dirname):
    """
    Load the label dict saved with a model
    """
    path = os.path.join(dirname, 'label-map.json')
    with open(path, 'r') as f:
        return json.load(f)


def load_params(dirname):
    """
    Load system parameters (not related to the networks)
    :return: a dictionary
    """
    path = os.path.join(dirname, 'system-params.json')
    with open(path, 'rb') as f:
        return json.load(f)


def read_alignment(filename, lowercase):
    """
    Read a file containing pairs of sentences and their alignments.
    :param filename: a JSONL file
    :param lowercase: whether to convert words to lowercase
    :return: a list of tuples (first_sent, second_sent, alignments)
    """
    sentences = []
    with open(filename, 'rb') as f:
        for line in f:
            line = line.decode('utf-8')
            if lowercase:
                line = line.lower()
            data = json.loads(line)
            sent1 = data['sentence1']
            sent2 = data['sentence2']
            alignment = data['alignment']
            sentences.append((sent1, sent2, alignment))

    return sentences


def read_corpuses(train_file, dev_file):
    return read_corpus(train_file), read_corpus(dev_file)


def read_corpus(filename):
    """
    Read a JSONL or TSV file with the SNLI corpus

    :param filename: path to the file
    :param lowercase: whether to convert content to lower case
    :param language: language to use tokenizer (only used if input is in
        TSV format)
    :return: a list of tuples (first_sent, second_sent, label)
    """
    print('Reading data from %s' % filename)
    # we are only interested in the actual sentences + gold label
    # the corpus files has a few more things
    useful_data = []

    # the SNLI corpus has one JSON object per line
    with open(filename, 'rb') as f:

        if filename.endswith('.tsv') or filename.endswith('.txt'):

            tokenize = tokenize_english
            for line in f:
                line = line.decode('utf-8').strip()
                line = line.lower()
                sent1, sent2, label = line.split('\t')
                if label == '-':
                    continue
                tokens1 = tokenize(sent1)
                tokens2 = tokenize(sent2)
                useful_data.append((tokens1, tokens2, label))
        else:
            for line in f:
                line = line.decode('utf-8')
                line = line.lower()
                data = json.loads(line)
                # gold label is the choosing the tag chosen by the majority (if there is one)
                if data['gold_label'] == '-':
                    # ignore items without a gold label
                    continue

                """
                Our first try to parse the data didn't work good enough, it seems there has to be a
                a splitting also the punctuation from words.
                #t = (data['sentence1'].split(" "), data['sentence2'].split(" "), data['gold_label'])
                """

                # taking the parsing tree of the sentences from the data
                sentence1_parse = data['sentence1_parse']
                sentence2_parse = data['sentence2_parse']
                label = data['gold_label']

                tree1 = nltk.Tree.fromstring(sentence1_parse)
                tree2 = nltk.Tree.fromstring(sentence2_parse)
                tokens1 = tree1.leaves()
                tokens2 = tree2.leaves()
                # tuple t contains the list of words from sentence 1 (tokens1) and from sentence 2 (tokens2).
                # and the label (which is the gold label)
                t = (tokens1, tokens2, label)
                useful_data.append(t)

    return useful_data
