import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import utils_part1 as utils
import sys

STUDENT = {'name': 'Raz Shenkman',
           'ID': '311130777'}

BATCH_SIZE = 1024
INPUT_SIZE = 250
LEARN_RATE = 0.06
EPOCHS = 3
EMBEDDING_VEC = 50
WINDOW = 5
HIDDEN_SIZE = 100


class ModelBuilder(object):
    def __init__(self):
        global HIDDEN_SIZE, EPOCHS, LEARN_RATE
        """
        The constructor initializes the datasets, model, optimizer and the dictionaries for the graph.
        """
        self.is_pos = False
        self.test_file = "ner/test"
        self.train_file = "ner/train"
        self.dev_file = "ner/dev"
        if len(sys.argv) == 0:
            print ("Usage tagger1.py pos/ner")
            exit()
        if sys.argv[1] == "pos":
            self.test_file = "pos/test"
            self.train_file = "pos/train"
            self.dev_file = "pos/dev"
            self.is_pos = True
            LEARN_RATE = 0.008
            HIDDEN_SIZE = 180
            EPOCHS = 4

        self.train_loader = utils.make_data_loader(self.train_file, batch_size=BATCH_SIZE)
        self.dev_loader = utils.make_data_loader(self.dev_file, dev=True)
        self.test_loader = utils.make_test_data_loader(self.test_file)
        self.model = FirstNet(input_size=INPUT_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LEARN_RATE)

        self.validation_loss_print_dict = {}
        self.validation_acc_print_dict = {}

    def train_validate_test(self):
        """
        This method trains the model and validate it, and then test the model & print the results
        into a graph.
        :return:
        """
        for epoch in range(1, EPOCHS + 1):
            self.train(epoch)
            self.validation(epoch)
        self.test()
        self.print_results_loss()
        self.print_results_acc()

    def print_results_loss(self):
        """
        This method draws the graph by using the validation epoch-loss
        :return:
        """
        norm_line, = plt.plot(self.validation_loss_print_dict.keys(), self.validation_loss_print_dict.values(), "red",
                              label='Validation loss')
        plt.legend(handler_map={norm_line: HandlerLine2D()})
        plt.show()

    def print_results_acc(self):
        """
        This method draws the graph by using the validation accuracy
        :return:
        """
        trained_line, = plt.plot(self.validation_acc_print_dict.keys(), self.validation_acc_print_dict.values(),
                                 "black",
                                 label='Validation accuracy')
        plt.legend(handler_map={trained_line: HandlerLine2D()})
        plt.show()

    def validation(self, epoch_num):
        """
        This method goes through the data, get the model output and validate it by using negative log likelihood loss
        also updates the dictionary and print the results.
        :param epoch_num: number of epoch we're in
        :return:
        """
        self.model.eval()
        validation_loss = 0
        correct = 0
        sum_examples = 0
        if self.is_pos:
            sum_examples = len(self.dev_loader)
        for data, target in self.dev_loader:
            output = self.model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            if self.is_pos:
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
            else:
                if utils.index_to_tags[target.cpu().sum().item()] == "O" and utils.index_to_tags[
                    pred.cpu().sum().item()] == "O":
                    continue
                sum_examples += 1
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        validation_loss /= len(self.dev_loader)
        print('\n Validation epoch number :{} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_num, validation_loss, correct, sum_examples,
            100. * correct / sum_examples))
        self.validation_loss_print_dict[epoch_num] = validation_loss
        self.validation_acc_print_dict[epoch_num] = float(100.0 * correct / sum_examples)

    def test(self):
        """
        This method goes through the data, get the model output create list of tags, later calls a
        function that tags the test file according to it.
        :return:
        """
        self.model.eval()
        preds = []
        for data in self.test_loader:
            output = self.model(torch.LongTensor(data))
            pred = output.data.max(1, keepdim=True)[1]
            preds.append(pred.cpu().sum().item())
        self.write_test_file(preds)

    def write_test_file(self, list_tags):
        """
        This method writes the test file according to the pos/ner flag
        :param list_tags: from the testing
        :return: new test file.
        """
        if self.is_pos:
            prediction_file = open("test1.pos", 'w')
        else:
            prediction_file = open("test1.ner", 'w')
        with open(self.test_file, "r") as original_test:
            i = 0
            for line in original_test.readlines():
                if line == "\n":
                    prediction_file.writelines("\n")
                    continue
                prediction_file.writelines(line.strip() + " " + str(utils.index_to_tags[list_tags[i]]) + "\n")
                i += 1

        prediction_file.close()

    def train(self, epoch):
        """
        This method goes through the data, get the model output and validate it by using negative log likelihood loss
        also train the model for every batch of examples we have looped through using our optimizer.
        :param epoch: number of epoch we're in
        :return:
        """
        self.model.train()
        correct = 0
        train_loss = 0
        total_examples = len(self.train_loader)
        for data, labels in self.train_loader:
            self.optimizer.zero_grad()
            output = self.model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(labels.data.view_as(pred)).cpu().sum().item()
            # negative log likelihood loss
            loss = F.nll_loss(output, labels)
            # calculate gradients
            train_loss += loss
            loss.backward()
            # update parameters
            self.optimizer.step()
        train_loss /= len(self.train_loader)
        print('\n Train epoch number: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch, train_loss, correct, total_examples * BATCH_SIZE,
                                        (100. * correct) / (total_examples * BATCH_SIZE)))


class FirstNet(nn.Module):
    """
    NN with 1 hidden layer that uses tanh.
    """

    def __init__(self, input_size):
        """
        Neural network inherits from nn.Module that has 1 hidden layer.
        """
        super(FirstNet, self).__init__()
        self.E = nn.Embedding(len(utils.words), EMBEDDING_VEC)
        self.input_size = WINDOW * EMBEDDING_VEC
        self.fc0 = nn.Linear(input_size, HIDDEN_SIZE)
        self.fc1 = nn.Linear(HIDDEN_SIZE, len(utils.tags))

    def forward(self, x):
        """
        For example x, get a vector of probabilities using softmax and the 1 hidden layer.
        :param x: example
        :return: vector of probabilities
        """

        x = self.E(x)
        x = x.view(-1, self.input_size)
        x = F.tanh(self.fc0(x))
        x = self.fc1(x)
        return F.log_softmax(x, dim=1)


def main():
    """
    Initializes model builder, train, validate and then test the model.
    :return:
    """
    my_obj = ModelBuilder()
    my_obj.train_validate_test()


if __name__ == "__main__":
    main()
