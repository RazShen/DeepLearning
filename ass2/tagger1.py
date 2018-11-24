from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import utils_part1 as utils


BATCH_SIZE = 50
INPUT_SIZE = 250
LEARN_RATE = 0.01
EPOCHS = 10
EMBEDDING_VEC = 50
WINDOW = 5


class ModelBuilder(object):
    def __init__(self):
        """
        The constructor initializes the datasets, model, optimizer and the dictionaries for the graph.
        """

        self.train_loader = utils.make_data_loader("pos/train", batch_size=BATCH_SIZE)
        self.dev_loader = utils.make_data_loader("pos/dev",dev=True)
        self.test_loader = utils.make_test_data_loader("pos/test")
        self.model = FirstNet(input_size=INPUT_SIZE)
        self.optimizer = optim.Adagrad(self.model.parameters(), lr=LEARN_RATE)
        #
        # train_dataset = utils
        # test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transforms.ToTensor())
        #
        # # Define the indices
        # indices = list(range(len(train_dataset)))  # start with all the indices in training set
        # split = int(len(train_dataset) * 0.2)  # define the split size
        #
        # # Random, non-contiguous split
        # validation_idx = np.random.choice(indices, size=split, replace=False)
        # train_idx = list(set(indices) - set(validation_idx))
        #
        # # define our samplers -- we use a SubsetRandomSampler because it will return
        # # a random subset of the split defined by the given indices without replacement
        # train_sampler = SubsetRandomSampler(train_idx)
        # validation_sampler = SubsetRandomSampler(validation_idx)
        #
        # # define loaders
        # self.train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        # self.validation_loader = DataLoader(dataset=train_dataset, batch_size=1, sampler=validation_sampler)
        # self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
        #
        # # initialize model

        #
        # # initialize optimizer for the model
        #
        # # initialize the dictionaries for the plot
        self.validation_print_dict = {}
        self.train_print_dict = {}

    def train_validate_test(self):
        """
        This method trains the model and validate it (10 epochs) and then test the model & print the results
        into a graph.
        :return:
        """
        for epoch in range(1, EPOCHS + 1):
            self.train(epoch)
            self.validation(epoch)
        self.test()
        self.print_results()

    def print_results(self):
        """
        This method draws the graph by using the validation and train epoch-loss dictionaries
        :return:
        """
        norm_line, = plt.plot(self.validation_print_dict.keys(), self.validation_print_dict.values(), "red",
                              label='Validation loss')
        trained_line, = plt.plot(self.train_print_dict.keys(), self.train_print_dict.values(), "black",
                                 label='Train loss')
        plt.legend(handler_map={norm_line: HandlerLine2D()})
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
        for data, target in self.dev_loader:
            output = self.model(data)
            validation_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        validation_loss /= len(self.dev_loader)
        print('\n Validation epoch number :{} Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            epoch_num, validation_loss, correct, len(self.dev_loader),
            100. * correct / len(self.dev_loader)))
        self.validation_print_dict[epoch_num] = validation_loss

    def test(self):
        """
        This method goes through the data, get the model output and validate it by using negative log likelihood loss
        also print the results for the test and write the predictions to the test.pred file.
        :return:
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        prediction_file = open("test.pred", 'w')
        for data, target in self.test_loader:
            output = self.model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            prediction_file.write(str(pred.item()) + "\n")
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        test_loss /= len(self.test_loader)
        print('\n Test Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader), 100. * correct / len(self.test_loader)))
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
            epoch, train_loss, correct, len(self.train_loader) * BATCH_SIZE,
                                        (100. * correct) / (len(self.train_loader) * BATCH_SIZE)))
        self.train_print_dict[epoch] = train_loss



class FirstNet(nn.Module):
    """
    Model A, Neural	Network	with two hidden	layers,	the first layer has	a size of 100 and the second layer has a size
    of 50, both are followed by	ReLU activation	function.
    """
    def __init__(self, input_size):
        """
        Neural network inherits from nn.Module that has 2 hidden layers, W1,b1,W2,b2,W3,b3.
        :param image_size: size of image
        """
        super(FirstNet, self).__init__()
        self.E = nn.Embedding(len(utils.words), EMBEDDING_VEC)
        self.input_size = WINDOW * EMBEDDING_VEC
        self.fc0 = nn.Linear(input_size, len(utils.tags))


    def forward(self, x):
        """
        For example x, get a vector of probabilities using softmax and the 1 hidden layer.
        :param x: example
        :return: vector of probabilities
        """
        #print(x)
        #print("&&&&&&&&&&&&&&&&&&&&&&NEXT X + " + str(len(x)) + "\n &&&&&&&&&&&&&&&&")

        x = self.E(x)
        x = x.view(-1, self.input_size)
        #print("&&&&&&&&&&&&&&&&&&&&&& after embedding      &&&&&&&&&&&&&&&&")
        x = F.tanh(self.fc0(x))
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