from itertools import izip
import numpy as np
start = '*_START_*'
end = '*_END_*'
unk = 'UUUNKKK'
word_embedding = {}
tags = set()
words = set()
word_to_index = {}
index_to_words = {}
tags_to_index = {}
index_to_tags = {}

def get_word_embedding(vec_file, words_file):
    global start, end
    word_embedding = {}
    for vec, word in izip(open(vec_file), open(words_file)):
        vec = str(vec)
        vec = vec.strip("\n").strip().split(" ")
        vec_numpy = np.asanyarray(map(float,vec))
        word = str(word)
        word = word.strip("\n").strip()
        word_embedding[word] = vec_numpy
    word_embedding[start] = np.random.uniform(-1, 1, [1, 50])
    word_embedding[end] = np.random.uniform(-1, 1, [1, 50])
    return word_embedding

def get_tagged_sentences(train_data_file):
    global tags, words
    tags = set()
    words = set()
    sentences = []
    curr_sentence = []
    with open(train_data_file, "r") as train_file:
        for word_and_tag in train_file.readlines():
            if word_and_tag == '\n':
                sentences.append(curr_sentence)
                curr_sentence = []
            else:
                word, tag = word_and_tag.strip("\n").split(" ")
                words.add(word)
                tags.add(tag)
                curr_sentence.append((word, tag))
    #tags = sorted(tags)
    #words = sorted(words)
    return sentences, tags, words



def make_window_of_words(sentences):
    global start, end, word_to_index
    all_windows = []
    all_tags = []
    for sentence in sentences:
        buffed_sentence = [(start, start), (start, start)]
        buffed_sentence.extend(sentence)
        buffed_sentence.extend([(end, end), (end, end)])

        for i, (word,tag) in enumerate(buffed_sentence[2:-2]):
            all_windows.append((word_to_index[buffed_sentence[i][0]],
                                word_to_index[buffed_sentence[i+1][0]], word_to_index[buffed_sentence[i+2][0]],
                                word_to_index[buffed_sentence[i+3][0]], word_to_index[buffed_sentence[i+4][0]]))
            all_tags.append(buffed_sentence[i+2][1])
    return all_windows, all_tags


def load_and_get_train_data(train_data_file):
    tagged_sentences, tags ,words = get_tagged_sentences(train_data_file)
    load_mapping_dicts(words, tags)
    window_of_words,all_tags = make_window_of_words(tagged_sentences)
    x= 5


def load_mapping_dicts(sorted_words, sorted_tags):
    global word_to_index, index_to_words, index_to_tags, tags_to_index
    word_to_index = {word: i + 2 for i, word in enumerate(list(sorted_words))}
    word_to_index[start] = 0
    word_to_index[end] = 1
    index_to_words = {i: word for word, i in word_to_index.iteritems()}
    tags_to_index = {tag: i for i, tag in enumerate(list(sorted_tags))}
    index_to_tags = {i: tag for tag, i in tags_to_index.iteritems()}


load_and_get_train_data("pos/train")
word_embedding = get_word_embedding("wordVectors.txt", "vocab.txt")

