import numpy as np
import loglinear as ll
STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    input = np.dot(params[2], np.tanh(np.dot(params[0], x)+b)) + params[3]
    probs = ll.softmax(input)
    return probs

def predict(x, params):
    """
    params: a list of the form [W, b, U, b_tag]
    """
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list of the form [W, b, U, b_tag]

    returns:
        loss,[gW, gb, gU, gb_tag]

    loss: scalar
    gW: matrix, gradients of W
    gb: vector, gradients of b
    gU: matrix, gradients of U
    gb_tag: vector, gradients of b_tag
    """
    # YOU CODE HERE
    W = params[0]
    b = params[1]
    U = params[2]
    b_tag = params[3]
    y_hat = predict(x,params)
    loss = -1 * np.log(y_hat[y])
    y_new = np.zeros(len(y_hat))
    y_new[y] = 1
    gb_tag = -(y_new - y_hat)
    gU = np.outer(gb_tag,np.tanh(np.dot(W,x) + b))
    gb = np.dot(np.dot(gb_tag, U), 1- np.power(np.tanh(np.dot(W,x) + b),2))
    gW = np.outer(gb ,x)
    return loss, [gW,gb,gU,gb_tag]

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    params = []
    W = np.zeros((hid_dim, in_dim))
    b = np.zeros(hid_dim)
    U = np.zeros((out_dim, hid_dim))
    b_tag = np.zeros(out_dim)
    params.append(W,b,U,b_tag)
    return params

