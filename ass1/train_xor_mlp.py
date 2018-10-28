import xor_data
import mlp1 as ml
import random
import numpy as np
import utils

STUDENT={'name': 'Raz Shenkman',
         'ID': '311130777'}


def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for y, x in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        y_hat = ml.predict(x,params)
        if y == y_hat:
            good += 1
        else:
            bad +=1
    return good / (good + bad)

def train_classifier(train_data, num_iterations, learning_rate, params):
    """
    Create and train a classifier, and return the parameters.

    train_data: a list of (label, feature) pairs.
    dev_data  : a list of (label, feature) pairs.
    num_iterations: the maximal number of training iterations.
    learning_rate: the learning rate to use.
    params: list of parameters (initial values)
    """

    for I in xrange(num_iterations):
        cum_loss = 0.0 # total loss in this iteration.
        for y, x in train_data:
            loss, grads = ml.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            for i,grad in enumerate(grads):
                params[i] -= learning_rate * grad
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        print I, train_loss, train_accuracy
    return params

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = np.random.randn(hid_dim, in_dim)
    b = np.random.randn(hid_dim)
    U = np.random.randn(out_dim, hid_dim)
    b_tag = np.random.randn(out_dim)
    params = [W,b,U,b_tag]
    return params

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    num_iterations = 30
    learning_rate = 0.03
    hidden = 16
    params = create_classifier(2, hidden, 2)
    trained_params = train_classifier(xor_data.data, num_iterations, learning_rate, params)
