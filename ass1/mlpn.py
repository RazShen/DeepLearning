import numpy as np
import loglinear as ll
STUDENT={'name': 'YOUR NAME',
         'ID': 'YOUR ID NUMBER'}


def classifier_output(x, params):
    # YOUR CODE HERE.
    h = x.copy()
    for W,b in zip(params[0:len(params)-1:2], params[1:len(params):2]):
        z = np.dot(h,W) + b
        h = np.tanh(z)
    probs = ll.softmax(z)
    return probs

def predict(x, params):
    return np.argmax(classifier_output(x, params))

def loss_and_gradients(x, y, params):
    """
    params: a list as created by create_classifier(...)

    returns:
        loss,[gW1, gb1, gW2, gb2, ...]

    loss: scalar
    gW1: matrix, gradients of W1
    gb1: vector, gradients of b1
    gW2: matrix, gradients of W2
    gb2: vector, gradients of b2
    ...

    (of course, if we request a linear classifier (ie, params is of length 2),
    you should not have gW2 and gb2.)
    """
    # YOU CODE HERE
    y_hat = classifier_output(x,params)
    loss = - np.log(y_hat[y])
    grad_saver = y_hat.copy()
    grad_saver[y] -= 1
    grads = []
    h = [x]
    """
    Calculate the h_i = tanh(z_i) = W_i*x + b_i from h_1 to h_n
    """
    for W, b in zip(params[0:-2:2], params[1:-1:2]):
        h.append(np.tanh(np.dot(h[-1], W) + b))

    """
    Calculate the gradients of each layer (start from the last layer)
    """
    for W, b in zip(params[-2::-2], params[-1::-2]):
        gb_i = np.copy(grad_saver)
        h_last = h.pop()
        gw_i = np.outer(h_last, grad_saver)
        grads.append(gb_i)
        grads.append(gw_i)
        grad_saver = np.dot(W, grad_saver) * (1 - np.square(np.tanh((h_last))))

    grads = list(reversed(grads))
    return loss, grads

def create_classifier(dims):
    """
    returns the parameters for a multi-layer perceptron with an arbitrary number
    of hidden layers.
    dims is a list of length at least 2, where the first item is the input
    dimension, the last item is the output dimension, and the ones in between
    are the hidden layers.
    For example, for:
        dims = [300, 20, 30, 40, 5]
    We will have input of 300 dimension, a hidden layer of 20 dimension, passed
    to a layer of 30 dimensions, passed to learn of 40 dimensions, and finally
    an output of 5 dimensions.
    
    Assume a tanh activation function between all the layers.

    return:
    a flat list of parameters where the first two elements are the W and b from input
    to first layer, then the second two are the matrix and vector from first to
    second layer, and so on.
    """
    params = []
    
    for first_dim,second_dim in zip(dims, dims[1:]):
        epsilon = np.sqrt(6) / (np.sqrt(first_dim + second_dim)) 
        params.append(np.random.uniform(-epsilon, epsilon, [first_dim, second_dim]))
        params.append(np.random.uniform(-epsilon, epsilon, [second_dim]))
    return params


