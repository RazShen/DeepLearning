import sys
import os
import dynet as dy
import random
import networks
import numpy
import pickle
from zipfile import ZipFile

tags = set()
words = set()
chars = set()
unk = "UUUNKKK"
word_to_index = {}
index_to_words = {}
tags_to_index = {}
index_to_tags = {}
chars_to_index = {}
index_to_chars = {}
prefixes = []
suffixes = []
prefixes_to_index = {}
suffixes_to_index = {}


def load_mapping_dicts():
    """
    load the dictionaries.
    :return:
    """
    global word_to_index, index_to_words, index_to_tags, tags_to_index, words, tags, chars, prefixes, \
        suffixes, prefixes_to_index, suffixes_to_index, chars_to_index, index_to_chars
    word_to_index = {word: i for i, word in enumerate(words)}
    index_to_words = {i: word for word, i in word_to_index.iteritems()}
    tags_to_index = {tag: i for i, tag in enumerate(tags)}
    index_to_tags = {i: tag for tag, i in tags_to_index.iteritems()}
    chars_to_index = {tag: i for i, tag in enumerate(chars)}
    index_to_chars = {i: tag for tag, i in chars_to_index.iteritems()}
    prefixes = list({index_to_words[i][: 3] for i in xrange(len(index_to_words))})
    suffixes = list({index_to_words[i][-3:] for i in xrange(len(index_to_words))})
    prefixes_to_index = {tag: i for i, tag in enumerate(prefixes)}
    suffixes_to_index = {tag: i for i, tag in enumerate(suffixes)}


def get_tagged_sentences(train_data_file, dev=False):
    """
    returns the sentences from the file with tags for it (as word and tuples)
    :param train_data_file:
    :param dev:
    :return:
    """
    global tags, words, unk, chars
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
                    chars.update(word)
                curr_sentence.append((word, tag))
    if not dev:
        words.add(unk)
        chars.add(unk)
    return sentences


def acc_on_dataset(dataset, network, is_pos):
    """
    Get accuracy on the data set, special case for NER
    :param dataset:
    :param network:
    :param is_pos:
    :return:
    """
    good = bad = 0.0
    for sentence in dataset:
        words_from_sentnce = [word for word, tag in sentence]
        tags_from_sentence = [tag for word, tag in sentence]
        prediction = network.get_tags_on_sentence(words_from_sentnce)
        for prediction, true_tag in zip(prediction, tags_from_sentence):
            if is_pos is False and true_tag == 'O' and prediction == 'O':
                continue
            if prediction == true_tag:
                good += 1
            else:
                bad += 1
    return good / (good + bad)


def save_model(pc, model_file):
    """
    save the model to a zip file with the name: model_file (inputted)
    :param pc:
    :param model_file:
    :return:
    """
    with open("model_dicts.pickle", "wb") as output:
        pickle.dump(word_to_index, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(tags_to_index, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(index_to_tags, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(chars_to_index, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(prefixes_to_index, output, pickle.HIGHEST_PROTOCOL)
        pickle.dump(suffixes_to_index, output, pickle.HIGHEST_PROTOCOL)
    pc.save("dynet_model.dy")
    with ZipFile(model_file, 'w') as zip:
        zip.write('model_dicts.pickle')
        zip.write('dynet_model.dy')
    os.remove("dynet_model.dy")
    os.remove("model_dicts.pickle")


def main(repr, train_file, model_file, type_pos_or_ner, dev_file=None, graph_name=None):
    """
    get the input from the train file, pick a network by the repr inputted, train for 5 epochs and then
    save the model and the dictionaries.
    :param repr:
    :param train_file:
    :param model_file:
    :param type_pos_or_ner:
    :param dev_file:
    :return:
    """
    tagged_sentences = get_tagged_sentences(train_file)
    is_pos = False
    if type_pos_or_ner == "pos":
        is_pos = True
    dev_sentences = []
    graph = {}
    if (dev_file):
        dev_sentences = get_tagged_sentences(dev_file, True)
    else:
        tagged_sentences, dev_sentences = tagged_sentences[:int((len(tagged_sentences) * 0.8))], \
                                          tagged_sentences[int(0.2 * len(tagged_sentences)):]
    load_mapping_dicts()
    pc = dy.ParameterCollection()
    if repr == 'a':
        network = networks.AEmbeddingBLSTM(pc, word_to_index, tags_to_index, index_to_tags)
    elif repr == 'b':
        network = networks.BCharBLSTM(pc, word_to_index, tags_to_index, chars_to_index, index_to_tags)
    elif repr == 'c':
        network = networks.CEmbeddingPrefSuffBLSTM(pc, word_to_index, tags_to_index, prefixes_to_index,
                                                   suffixes_to_index, index_to_tags)
    else:
        network = networks.ABConcatBLSTM(pc, word_to_index, tags_to_index, chars_to_index, index_to_tags)

    if repr == 'd':
        trainer = dy.AdamTrainer(pc, 0.00015)
    else:
        trainer = dy.AdamTrainer(pc, 0.0004)

    # total_seen_sentences = 0
    for epoch in xrange(5):
        losses = []
        dev_accs = []
        iteration = 0
        random.shuffle(tagged_sentences)
        for sentence in tagged_sentences:
            # total_seen_sentences += 1
            words = [word for word, tag in sentence]
            tags = [tag for word, tag in sentence]
            loss = network.get_loss_on_sentence(words, tags)
            losses.append(loss.value())  # need to run loss.value() for the forward prop
            loss.backward()
            trainer.update()
            iteration += 1
            if iteration % 500 == 0:
                avg = numpy.average(losses)
                losses = []
                dev_acc = acc_on_dataset(dev_sentences, network, is_pos)
                # graph[int(total_seen_sentences/100)] = dev_acc
                dev_accs.append(dev_acc)
                print("dev acc: " + str(dev_acc))
        print 'Finished epoch: ' + str(epoch + 1)
    save_model(pc, model_file)
    with open(graph_name, "wb") as output:
        pickle.dump(graph, output, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main(*sys.argv[1:])
