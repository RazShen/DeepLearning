from __future__ import division, unicode_literals
import json
import numpy as np
import nltk
from collections import defaultdict, Counter

tokenizer_utils = nltk.tokenize.TreebankWordTokenizer()

UNKNOWN = '**UNK**'
PADDING = '**PAD**'
START = '**START**'  # to be inserted in the start of each sentence


class DatasetComparison(object):
    """
    dataset object for SNLI model
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
        Shuffle the dataset order.
        :return:
        """
        shuffle_arbitrary_num_of_arrays(self.sentences1, self.sentences2, self.sizes1, self.sizes2, self.labels)

    def get_batch_from_range(self, from_, to):
        """
        Return DataComparison object of a given range from the dataset ( [from_ : to] )
        :param from_: start index
        :param to: end index
        :return: DataComparison object
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
        Mapps all lables from tagged pairs (sentence1, sentence2, label) to integers
        :param data pairs: tagged pairs : list of (sentence1, sentence2, label)
        :return: a mapping dict
        """
        labels = set(pair[2] for pair in pairs)
        return dict(zip(labels, range(len(labels))))


    def labels_to_np_arrays(pairs, label_map):
        """
        Return np array containing the numerical representation of the labels in pairs.
        :param pairs: tagged pairs : (sentence1, sentence2, label)
        :param label_map: a mapping dict of labels strings to integers
        :return: labels' np array
        """
        return np.array([label_map[pair[2]] for pair in pairs], dtype=np.int32)


    def create_dataset(pairs, word_dict, label_dict=None, max_len1=None, max_len2=None):
        """
        Creates a ComparisonDataset from the given pairs.
        :param pairs: tagged pairs : (sentence1, sentence2, label)
        :param word_dict: mapping every word to its embedding vector index
        :param label_dict: a mapping dict of labels strings to integers
        :param max_len1: the maximum length of sentence from all the first sentences
        :param max_len2: the maximum length of sentence from all the first sentences
        :return: ComparisonDataset
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
        Converts all words in the the sentences to numpy matrix with the words' indices.
        In this step we add the START symbol at the beginning of each sentence.
        :param sentences: list of sentences, each sentence is a list of tokens
        :param word_dict: mapping every word to its embedding vector index
        :param max_len: maximum length allowed for sentence. If not given - then max)len will be the length
        of the longest sentence
        :return: numpy matrix with the words' indices and numpy array with the sentences lengths
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
        Loads the word embeddings text file and return:
        list of all the words in the file, and numpy embedding matrix.
        The embed vector in the i-th row belongs to the i-th word in the words list
        :param path: path to words embeddings file
        :return: words list, embedding matrix
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


    def load_corpus(filename):
        """
        Reads .jsonl file of SNLI dataset and returns list of tuples, each tuple is an example from the dataset and contains:
        (sentence1, sentence2, label)
        :param filename: path to the data file
        :return: list of tuples, each tuple is an example from the dataset and contains:
        (sentence1, sentence2, label)
        """
        print("Reading the file " + filename)

        with open(filename, 'rb') as f:

            if filename.endswith('.tsv') or filename.endswith('.txt'):

                tokenize = tokenize_english
                for line in f:
                    line = line.decode('utf-8').strip().lower()
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
                        # don't use examples without gold label
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
