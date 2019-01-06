from __future__ import division, unicode_literals
import json
import numpy as np
import nltk
from collections import defaultdict, Counter

tokenizer_utils = nltk.tokenize.TreebankWordTokenizer()

UNKNOWN = '**UNK**'
PADDING = '**PAD**'
START = '**START**'  # it's called "START" but actually serves as a null alignment


class DatasetComparison(object):
    """
    Class for better organizing a data set. It provides a separation between
       first and second sentences and also their sizes.
    """

    def __init__(self, sentences1, sentences2, sizes1, sizes2, labels):
        """
        :param sentences1: A 2D numpy array with sentences (the first in each
            pair) composed of token indices
        :param sentences2: A 2D numpy array with sentences (the second in each
            pair) composed of token indices
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

    def shuffle_sentences(self):
        """
        Shuffle all data using the same random sequence.
        :return:
        """
        shuffle_arbitrary_num_of_arrays(self.sentences1, self.sentences2, self.sizes1, self.sizes2, self.labels)

    def get_batch_from_range(self, from_, to):
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

        subset = DatasetComparison(self.sentences1[from_:to], self.sentences2[from_:to], self.sizes1[from_:to],
                                   self.sizes2[from_:to], self.labels[from_:to])
        return subset


def tokenize_english(text):
    """
    Tokenize a piece of text using the Treebank tokenizer

    :return: a list of strings
    """
    return tokenizer_utils.tokenize(text)


def shuffle_arbitrary_num_of_arrays(*arrays):
    """
    Shuffle all given arrays with the same RNG state.

    All shuffling is in-place, i.e., this function returns None.
    """
    rng_state = np.random.get_state()
    for array in arrays:
        np.random.shuffle(array)
        np.random.set_state(rng_state)


def get_label_dictionary(pairs):
    """
    Return a dictionary mapping the labels found in `pairs` to numbers
    :param pairs: a list of tuples (_, _, label), with label as a string
    :return: a dict
    """
    labels = set(pair[2] for pair in pairs)
    return dict(zip(labels, range(len(labels))))


def labels_to_np_arrays(pairs, label_map):
    """
    Return a numpy array representing the labels in `pairs`

    :param pairs: a list of tuples (_, _, label), with label as a string
    :param label_map: dictionary mapping label strings to numbers
    :return: a numpy array
    """
    return np.array([label_map[pair[2]] for pair in pairs], dtype=np.int32)


def create_dataset(pairs, word_dict, label_dict=None, max_len1=None, max_len2=None):
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
    tkn1 = [pair[0] for pair in pairs]
    tkn2 = [pair[1] for pair in pairs]
    sen1, sizes1 = get_pairs_to_indices(tkn1, word_dict, max_len1)
    sen2, sizes2 = get_pairs_to_indices(tkn2, word_dict, max_len2)
    if label_dict is not None:
        labels = labels_to_np_arrays(pairs, label_dict)
        return DatasetComparison(sen1, sen2, sizes1, sizes2, labels)
    else:
        labels = None
        return DatasetComparison(sen1, sen2, sizes1, sizes2, labels)


def get_pairs_to_indices(sentences, word_dict, max_len=None):
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
    lenths = np.array([len(sent) for sent in sentences])
    lenths += 1
    if max_len is not None:
        max_len += 1
    else:
        max_len = lenths.max()
    # max_len is the length of the longest sentence if None
    shape = (len(sentences), max_len)
    # make matrix of number_of_sentences X maximum_length_of_sentence where each value is index of PAD word.
    padded_matrix = np.full(shape, word_dict[PADDING], dtype=np.int32)
    # Fill the array with the actual indices up to the length of each sentence.
    for i, sent in enumerate(sentences):
        indices = [word_dict[token] for token in sent]
        indices = [word_dict[START]] + indices
        padded_matrix[i, :len(indices)] = indices

    return padded_matrix, lenths


def get_sentence_sizes(pairs):
    """
    Count the sizes of all sentences in the pairs
    :param pairs: a list of tuples (sent1, sent2, _). They must be
        tokenized
    :return: a tuple (sizes1, sizes2), as two numpy arrays
    """
    size_sen1 = np.array([len(pair[0]) for pair in pairs])
    size_sen2 = np.array([len(pair[1]) for pair in pairs])
    return (size_sen1, size_sen2)


def get_random_vec(size):
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
    :return: a tuple (defaultdict, array)
    """

    print('Getting Embedding')
    wordlist, embeds = load_text_embeddings(embeddings_path)

    # mapping every word to tuple of (word, index) where the indices starts from 3 up to len(wordlist) + 3
    # saving indices 0-2 for special cases.
    mapping = zip(wordlist, range(3, len(wordlist) + 3))

    # always map OOV words to 0
    word_embeds = defaultdict(int, mapping)
    word_embeds[UNKNOWN] = 0
    word_embeds[PADDING] = 1
    word_embeds[START] = 2
    # generating 3 random vectors for unknown, padding, start
    vector_size = embeds.shape[1]
    extra = [get_random_vec(vector_size), get_random_vec(vector_size), get_random_vec(vector_size)]
    embeds = np.append(extra, embeds, 0)
    embeds = embeds / np.linalg.norm(embeds, axis=1).reshape((-1, 1))
    # wd is a dictionary that maps from word to index
    return word_embeds, embeds


def load_text_embeddings(path):
    """
    Load any embedding model written as text, in the format:
    word[space or tab][values separated by space or tab]

    :param path: path to embeddings file
    :return: a tuple (wordlist, array)
    """
    words = []
    # start from index 1 and reserve 0 for unknown
    word_vecs = []
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
            word_vecs.append(vector)  # vectors is list of numpy arrays
    # make embeddings as numpy array (numpy array of numpy arrays)
    embeddings = np.array(word_vecs, dtype=np.float32)
    return words, embeddings


def read_corpuses(train_file, test_file):
    return read_corpus(train_file), read_corpus(test_file)


def read_corpus(filename):
    """
    Read a JSONL or TSV file with the SNLI corpus

    :param filename: path to the file
    :return: a list of tuples (first_sent, second_sent, label)
    """
    print("Reading the file " + filename)
    # we are only interested in the actual sentences + gold label
    # the corpus files has a few more things
    useful_data = []

    # the SNLI corpus has one JSON object per line
    with open(filename, 'rb') as f:

        if filename.endswith('.tsv') or filename.endswith('.txt'):

            tokenize = tokenize_english
            for line in f:
                line = line.decode('utf-8').strip().lower()
                sent1, sent2, label = line.split('\t')
                if label == '-':
                    continue
                tkn1 = tokenize(sent1)
                tkn2 = tokenize(sent2)
                useful_data.append((tkn1, tkn2, label))
        else:
            for line in f:
                line = line.decode('utf-8').lower()
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
                tkn1 = tree1.leaves()
                tkn2 = tree2.leaves()
                # tuple t contains the list of words from sentence 1 (tokens1) and from sentence 2 (tokens2).
                # and the label (which is the gold label)
                t = (tkn1, tkn2, label)
                useful_data.append(t)

    return useful_data
