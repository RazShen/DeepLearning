from __future__ import division, print_function
import tensorflow as tf
import ioutils
import utils
from multimlp import MultiFeedForwardClassifier

def main():
    embedding_file = "glove.6B.200d.txt"
    train_file = "snli_1.0/snli_1.0_train.jsonl"
    validation_file = "snli_1.0/snli_1.0_dev.jsonl"
    save_path = "saved_model"
    hidden_units = 100

    # train pairs is the list of tuples ([sentence1 words], [sentence2 words], label)
    train_pairs = ioutils.read_corpus(train_file)
    # validation pairs is the list of tuples ([sentence1 words], [sentence2 words], label)
    valid_pairs = ioutils.read_corpus(validation_file)

    # whether to generate embeddings for unknown, padding, null
    # word_dict is a dictionary of words and indices and embeddings matrix is a matrix where its indices are the
    # vector for the word...
    word_dict, embeddings = ioutils.load_embeddings(embedding_file)

    print('Converting words to indices')
    # find out which labels are there in the data
    # (more flexible to different datasets)
    label_dict = utils.create_label_dict(train_pairs)  # create dictionary of the labels and their index
    # train_pairs is list of tuples, where each tuple is ([sentence1 words], [sentence2 words], label).
    train_data = utils.create_dataset(train_pairs, word_dict, label_dict)
    valid_data = utils.create_dataset(valid_pairs, word_dict, label_dict)

    sess = tf.InteractiveSession()
    print('Creating model')
    vocab_size = embeddings.shape[0]
    embedding_size = embeddings.shape[1]

    model = MultiFeedForwardClassifier(vocab_size, embedding_size)

    model.initialize(sess, embeddings)
    print('Starting training')
    model.train(sess, train_data, valid_data)


if __name__ == '__main__':
    main()