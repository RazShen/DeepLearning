import numpy as np
import loglinear as ll
import math
STUDENT={'name': 'Raz Shenkman',
         'ID': '311130777'}

def classifier_output(x, params):
    # YOUR CODE HERE.
    W, b, U, b_tag = params

    probs = ll.softmax(np.dot(U,(np.tanh(np.dot(W,x)+b)))+b_tag)
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
    W, b, U, b_tag = params
    y_hat = classifier_output(x,params)
    loss = -np.log(y_hat[y])

    gb_tag = y_hat.copy()
    gb_tag[y] -= 1
    hidden_output = np.tanh(np.dot(W,x) + b)
    
    gU = np.outer(gb_tag,hidden_output)

    gb = np.dot(gb_tag, U) * (1 - np.square(hidden_output))

    gW = np.outer(gb ,x)

    return loss, [gW,gb,gU,gb_tag]


def xavier_vec(dim1):
    epsilon = np.sqrt(6) / (np.sqrt(dim1))
    return np.random.uniform(-epsilon, epsilon, dim1)


def xavier_mat(dim1, dim2):
    epsilon = np.sqrt(6) / (np.sqrt(dim1 + dim2))
    return np.random.uniform(-epsilon, epsilon, [dim1, dim2]) 

def create_classifier(in_dim, hid_dim, out_dim):
    """
    returns the parameters for a multi-layer perceptron,
    with input dimension in_dim, hidden dimension hid_dim,
    and output dimension out_dim.

    return:
    a flat list of 4 elements, W, b, U, b_tag.
    """
    W = xavier_mat(hid_dim, in_dim)
    b = xavier_vec(hid_dim)
    U = xavier_mat(out_dim, hid_dim)
    b_tag = xavier_vec(out_dim)
    params = [W,b,U,b_tag]
    return params


 
