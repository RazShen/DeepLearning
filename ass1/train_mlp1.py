import mlp1 as ml
import random
import numpy as np
import utils

STUDENT={'name': 'Raz Shenkman',
         'ID': '311130777'}

def feats_to_vec(features):
    # YOUR CODE HERE.
    # Should return a numpy vector of features.
    vec = np.zeros(len(utils.vocab))
    for bigram in features:
        if bigram in utils.vocab:
            vec[utils.F2I[bigram]] += 1
    total = sum(vec)
    vec = np.divide(vec, total)
    return vec

def accuracy_on_dataset(dataset, params):
    good = bad = 0.0
    for label, features in dataset:
        # YOUR CODE HERE
        # Compute the accuracy (a scalar) of the current parameters
        # on the dataset.
        # accuracy is (correct_predictions / all_predictions)
        x = feats_to_vec(features) # convert features to a vector.
        y = utils.L2I[label]       # convert the label to number if needed.
        y_hat = ml.predict(x,params)
        if y == y_hat:
            good += 1
        else:
            bad +=1
    return good / (good + bad)

def train_classifier(train_data, dev_data, num_iterations, learning_rate, params):
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
        random.shuffle(train_data)
        for label, features in train_data:
            x = feats_to_vec(features) # convert features to a vector.
            y = utils.L2I[label]                  # convert the label to number if needed.
            loss, grads = ml.loss_and_gradients(x,y,params)
            cum_loss += loss
            # YOUR CODE HERE
            # update the parameters according to the gradients
            # and the learning rate.
            for i,grad in enumerate(grads):
                params[i] -= learning_rate * grad
        train_loss = cum_loss / len(train_data)
        train_accuracy = accuracy_on_dataset(train_data, params)
        dev_accuracy = accuracy_on_dataset(dev_data, params)
        print I, train_loss, train_accuracy, dev_accuracy
    return params

def pred_on_test(trained_params, test):
    prediction_file = open("test.pred", 'w')
    for q, features in test:
        x = feats_to_vec(features) # convert features to a vector.
        y_hat = ml.predict(x,params)
        y_hat_name_of_lang = [key for key, value in utils.L2I.iteritems() if value == y_hat][0]
        prediction_file.write(str(y_hat_name_of_lang) + "\n")
    prediction_file.close()

if __name__ == '__main__':
    # YOUR CODE HERE
    # write code to load the train and dev sets, set up whatever you need,
    # and call train_classifier.
    
    num_iterations = 30
    learning_rate = 0.03
    hidden = 20
    in_dim = len(utils.vocab)
    out_dim = len(utils.L2I)
    params = ml.create_classifier(in_dim, out_dim, hidden)
    trained_params = train_classifier(utils.TRAIN, utils.DEV, num_iterations, learning_rate, params)
    pred_on_test(trained_params, utils.TEST)
