import dynet as dy
import sys
import utils_part_1 as utils
import random
import numpy as np
from time import time

STUDENT = {'name': 'Raz Shenkman',
           'ID': '311130777'}


# acceptor LSTM
class LstmAcceptor(object):
    def __init__(self, in_dim, lstm_dim, out_dim, model):
        self.builder = dy.VanillaLSTMBuilder(1, in_dim, lstm_dim, model)
        self.W = model.add_parameters((out_dim, lstm_dim))

    def __call__(self, sequence):
        lstm = self.builder.initial_state()
        W = self.W.expr()  # convert the parameter into an Expession (add it to graph)
        outputs = lstm.transduce(sequence)
        result = W * outputs[-1]
        return result


class Acceptor(object):
    def __init__(self, vocab, embed_dim, lstm_dim, out_dim):
        """
        The constructor initializes the datasets, model, optimizer and the dictionaries for the graph.
        """
        self.train_file_path = sys.argv[1]
        self.test_file_path = sys.argv[2]
        self.train_examples = utils.read_examples_file(self.train_file_path)
        self.test_examples = utils.read_examples_file(self.test_file_path)
        self.m = dy.Model()
        self.trainer = dy.AdamTrainer(self.m)
        self.E = self.m.add_lookup_parameters((len(vocab), embed_dim))
        self.acceptor = LstmAcceptor(embed_dim, lstm_dim, out_dim, self.m)
        self.v2i = {c: i for i, c in enumerate(vocab)}

    def train(self, epochs):
        random.shuffle(self.train_examples)
        start_time = time()
        for epoch in range(1, epochs + 1):
            losses = 0.0
            for example, tag in self.train_examples:
                dy.renew_cg()
                vecs = [self.E[self.v2i[letter]] for letter in example]
                preds = self.acceptor(vecs)
                loss = dy.pickneglogsoftmax(preds, tag)
                losses += loss.value()
                loss.backward()
                self.trainer.update()
            train_acc = self.accuracy_on_data_set(self.train_examples)
            test_acc = self.accuracy_on_data_set(self.test_examples)
            test_loss = self.get_train_loss()
            print ("epoch number: " + str(epoch) + ", train accuracy: " + str(100.0 * train_acc) + ", train loss: "
                   + str(float(losses) / len(self.train_examples)) + ", test acc: " + str(100.0 * test_acc)
                   + ", test loss: " + str(test_loss))
        print ("Train time took:" + str(time() - start_time))

    def get_train_loss(self):
        losses = 0.0
        for example, tag in self.test_examples:
            dy.renew_cg()
            vecs = [self.E[self.v2i[letter]] for letter in example]
            preds = self.acceptor(vecs)
            loss = dy.pickneglogsoftmax(preds, tag)
            losses += loss.value()
        return losses/len(self.test_examples)

    def predict_on_example(self, example):
        dy.renew_cg()  # new computation graph
        vecs = [self.E[self.v2i[c]] for c in example]
        preds = dy.softmax(self.acceptor(vecs))
        vals = preds.npvalue()
        pred = np.argmax(vals)
        return pred

    def accuracy_on_data_set(self, dataset):
        tp_or_fn = 0.0
        fp_or_tn = 0.0
        for example, tag in dataset:
            pred = self.predict_on_example(example)
            if pred == tag:
                tp_or_fn += 1
            else:
                fp_or_tn += 1
        return tp_or_fn / (tp_or_fn + fp_or_tn)


def main():
    """
    Initializes model builder, train, validate and then test the model.
    :return:
    """
    if len(sys.argv) < 3:
        print ("Usage python experiment.py <train file path> <test file path>")
        exit()
    vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d"]
    vocab2 = ["%","a", "b", "c", "d", "e", "f","g","h", "i", "j", "k", "l", "m","n","o","p", "q","r","s","t","u","v"
              ,"w","x", "y", "z"]
    epochs = 10
    vec_per_char_size = 100
    lstm_dim = 100
    output_dim = 2
    my_acceptor = Acceptor(vocab2, vec_per_char_size, lstm_dim, output_dim)
    my_acceptor.train(epochs)


if __name__ == "__main__":
    main()
