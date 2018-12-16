import networks
import sys
import pickle
import dynet as dy
from zipfile import ZipFile
import os

word_to_index = {}
tags_to_index = {}
index_to_tags = {}
chars_to_index = {}
prefixes_to_index = {}
suffixes_to_index = {}


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


def load_dicts(model_file):
    """
    Load the dictionaries from the saved model zip
    :param model_file: zip of the saved model
    :return:
    """
    global word_to_index, tags_to_index, index_to_tags, chars_to_index, prefixes_to_index, suffixes_to_index
    with ZipFile(model_file) as model_zip:
        model_zip.extractall(os.getcwd())
    with open("model_dicts.pickle") as dicts:
        word_to_index = pickle.load(dicts)
        tags_to_index = pickle.load(dicts)
        index_to_tags = pickle.load(dicts)
        chars_to_index = pickle.load(dicts)
        prefixes_to_index = pickle.load(dicts)
        suffixes_to_index = pickle.load(dicts)
    os.remove("model_dicts.pickle")


def main(repr, model_file, file_to_tag, is_pos):
    pos = False
    if is_pos == "pos":
        pos = True
    pc = dy.ParameterCollection()
    load_dicts(model_file)
    if repr == 'a':
        network = networks.AEmbeddingBLSTM(pc, word_to_index, tags_to_index, index_to_tags)
    elif repr == 'b':
        network = networks.BCharBLSTM(pc, word_to_index, tags_to_index, chars_to_index, index_to_tags)
    elif repr == 'c':
        network = networks.CEmbeddingPrefSuffBLSTM(pc, word_to_index, tags_to_index, prefixes_to_index, suffixes_to_index,
                                                   index_to_tags)
    else:
        network = networks.ABConcatBLSTM(pc, word_to_index, tags_to_index, chars_to_index, index_to_tags)
    # setup dynet model from the extracted zip model
    pc.populate("dynet_model.dy")
    os.remove("dynet_model.dy")

    sentence = get_not_tagged_sentences(file_to_tag)
    final_tags = []
    for sentence in sentence:
        tags = network.get_tags_on_sentence(sentence)
        final_tags.extend(tags)
    write_test_file(pos, final_tags, file_to_tag)


def write_test_file(is_pos, list_tags, test_file):
    """
    Writing the test file using the original input file
    :param is_pos:
    :param list_tags:
    :param test_file:
    :return:
    """
    if is_pos:
        prediction_file = open("test4.pos", 'w')
    else:
        prediction_file = open("test4.ner", 'w')
    with open(test_file, "r") as original_test:
        i = 0
        for line in original_test:
            if line == "\n":
                prediction_file.write("\n")
                continue
            prediction_file.write(line.strip() + " " + str(list_tags[i]) + "\n")
            i += 1
    prediction_file.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
