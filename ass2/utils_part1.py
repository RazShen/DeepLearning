from itertools import izip
import numpy as np
import torch
import torch.utils.data

start = '*_START_*'
end = '*_END_*'
unk = "UUUNKKK"
word_embedding = {}
tags = set()
words = set()
word_to_index = {}
index_to_words = {}
tags_to_index = {}
index_to_tags = {}
EMBEDDING_VEC_SIZE = 50

STUDENT = {'name': 'Raz Shenkman',
           'ID': '311130777'}


def get_word_embedding(vec_file, words_file):
    """
    Returns the word embedding dictionary.
    :param vec_file: vectors
    :param words_file: words
    :return:
    """
    global start, end
    word_embedding = {}
    for vec, word in izip(open(vec_file), open(words_file)):
        vec = str(vec)
        vec = vec.strip("\n").strip().split(" ")
        vec_numpy = np.asanyarray(map(float, vec))
        word = str(word)
        word = word.strip("\n").strip()
        word_embedding[word] = vec_numpy
    return word_embedding


def get_tagged_sentences(train_data_file, dev=False):
    """
    returns the sentences from the file with tags for it (as word and tuples)
    :param train_data_file:
    :param dev:
    :return:
    """
    global tags, words, unk
    sentences = []
    curr_sentence = []
    with open(train_data_file, "r") as train_file:
        for word_and_tag in train_file.readlines():
            if word_and_tag == '\n':
                sentences.append(curr_sentence)
                curr_sentence = []
            else:
                word, tag = word_and_tag.strip("\n").strip().strip("\t").split()
                if not dev:
                    words.add(word)
                    tags.add(tag)
                curr_sentence.append((word, tag))
    if not dev:
        words.add(unk)
    return sentences


def get_not_tagged_sentences(test_data_file):
    """
    returns the sentence from the file w/o tags (only words)
    :param test_data_file:
    :return:
    """
    sentences = []
    curr_sentence = []
    with open(test_data_file, "r") as train_file:
        for word_in_line in train_file.readlines():
            if word_in_line == '\n':
                sentences.append(curr_sentence)
                curr_sentence = []
            else:
                word = word_in_line.strip("\n").strip()
                curr_sentence.append(word)
    return sentences


def make_window_of_words(sentences):
    """
    returns window of words in sentences
    :param sentences: list of windows (each window is index of words)
    :return:
    """
    global start, end, word_to_index
    all_windows = []
    all_tags = []
    for sentence in sentences:
        buffed_sentence = [(start, start), (start, start)]
        buffed_sentence.extend(sentence)
        buffed_sentence.extend([(end, end), (end, end)])
        for i, (word, tag) in enumerate(buffed_sentence):
            if word == start or word == end:
                continue
            all_windows.append(get_indexes_of_windows_from_test(buffed_sentence[i - 2][0], buffed_sentence[i - 1][0],
                                                                buffed_sentence[i][0], buffed_sentence[i + 1][0],
                                                                buffed_sentence[i + 2][0]))
            all_tags.append(tags_to_index[buffed_sentence[i][1]])
    return all_windows, all_tags


def make_window_of_untagged_words(sentences):
    """
    returns window of words in sentences
    :param sentences: list of windows (each window is index of words)
    :return:
    """
    global start, end
    all_windows = []
    for sentence in sentences:
        buffed_sentence = [start, start]
        buffed_sentence.extend(sentence)
        buffed_sentence.extend([end, end])
        for i, word in enumerate(buffed_sentence):
            if word == start or word == end:
                continue
            all_windows.append(get_indexes_of_windows_from_test(buffed_sentence[i - 2], buffed_sentence[i - 1],
                                                                buffed_sentence[i], buffed_sentence[i + 1],
                                                                buffed_sentence[i + 2]))

    return all_windows


def get_indexes_of_windows_from_test(w1, w2, w3, w4, w5):
    """
    Gets words and return a list of their indexes
    """
    final_win = [get_word_index(w1), get_word_index(w2), get_word_index(w3), get_word_index(w4), get_word_index(w5)]
    return final_win


def get_word_index(word):
    """
    return the index of the word if its indexed, otherwise unk index
    :param word:
    :return:
    """
    global unk, word_to_index
    if word in word_to_index:
        return word_to_index[word]
    else:
        return word_to_index[unk]


def load_and_get_train_data(train_data_file, dev=False):
    """
    get windows of indexes of words from the train data.
    :param test_data_file:
    :return:
    """
    tagged_sentences = get_tagged_sentences(train_data_file, dev)
    if not dev:
        load_mapping_dicts()
    window_of_words, all_tags = make_window_of_words(tagged_sentences)
    return window_of_words, all_tags


def load_and_get_test_data(test_data_file):
    """
    get windows of indexes of words from the test data.
    :param test_data_file:
    :return:
    """
    sentences = get_not_tagged_sentences(test_data_file)
    window_of_words = make_window_of_untagged_words(sentences)
    return window_of_words


def make_test_data_loader(file, batch_size=1):
    """
    make the windows for the test
    :param file:
    :param batch_size:
    :return:
    """
    windows = load_and_get_test_data(file)
    return windows


def make_data_loader(file, dev=False, batch_size=1):
    """
    make data loaders for train and validation
    :param file:
    :param dev:
    :param batch_size:
    :return:
    """
    windows, tags_for_windows = load_and_get_train_data(file, dev)
    windows, tags_for_windows = torch.from_numpy(np.array(windows)), torch.from_numpy(np.array(tags_for_windows))
    windows, tags_for_windows = windows.type(torch.LongTensor), tags_for_windows.type(torch.LongTensor)
    data = torch.utils.data.TensorDataset(windows, tags_for_windows)
    return torch.utils.data.DataLoader(data, batch_size, shuffle=True)


def load_mapping_dicts():
    """
    load the dictionaries.
    :return:
    """
    global word_to_index, index_to_words, index_to_tags, tags_to_index, start, end, words, tags
    words.update({start, end})
    word_to_index = {word: i for i, word in enumerate(words)}
    index_to_words = {i: word for word, i in word_to_index.iteritems()}

    tags_to_index = {tag: i for i, tag in enumerate(tags)}
    index_to_tags = {i: tag for tag, i in tags_to_index.iteritems()}
