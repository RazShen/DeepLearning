from __future__ import division, print_function
import tensorflow as tf
import utils
from SNLIModel import SNLIModel


def main():
    # data file paths
    embedding_file = "glove.6B.200d.txt"
    train_file = "snli_1.0/snli_1.0_train.jsonl"
    test_file = "snli_1.0/snli_1.0_test.jsonl"

    # train pairs is the list of tuples ([sentence1 words], [sentence2 words], label)
    # test pairs is the list of tuples ([sentence1 words], [sentence2 words], label)
    train_pairs, test_pairs = utils.read_corpuses(train_file, test_file)

    # word_dict is a dictionary of words and indices and embeddings matrix is a matrix where its indices are the
    # vector for the word...
    word_dict, embeddings = utils.load_embeddings(embedding_file)

    # find out which labels are there in the data
    # (more flexible to different datasets)
    label_dict = utils.get_label_dictionary(train_pairs)  # create dictionary of the labels and their index
    # train_pairs is list of tuples, where each tuple is ([sentence1 words], [sentence2 words], label).
    train_data = utils.create_dataset(train_pairs, word_dict, label_dict)
    test_data = utils.create_dataset(test_pairs, word_dict, label_dict)
    # setup interactive session to run the computations in
    sess = tf.InteractiveSession()
    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]
    size_of_network = 100
    snli_model = SNLIModel(size_of_network, vocab_size, embedding_size)

    snli_model.init_tf_var(sess, embeddings)
    print('Training part')
    snli_model.train(sess, train_data, test_data)


if __name__ == '__main__':
    main()
