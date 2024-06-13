"""
neuralnet.py

What you need to do:
- Complete random_init
- Implement SoftMaxCrossEntropy methods
- Implement Sigmoid methods
- Implement Linear methods
- Implement NN methods

It is ***strongly advised*** that you finish the Written portion -- at the
very least, problems 1 and 2 -- before you attempt this programming 
assignment; the code for forward and backprop relies heavily on the formulas
you derive in those problems.

Sidenote: We annotate our functions and methods with type hints, which
specify the types of the parameters and the returns. For more on the type
hinting syntax, see https://docs.python.org/3/library/typing.html.
"""

import numpy as np
import argparse
from typing import Callable, List, Tuple

# This takes care of command line argument parsing for you!
# To access a specific argument, simply access args.<argument name>.
parser = argparse.ArgumentParser()
parser.add_argument('train_input', type=str,
                    help='path to training input .csv file')
parser.add_argument('validation_input', type=str,
                    help='path to validation input .csv file')
parser.add_argument('train_out', type=str,
                    help='path to store prediction on training data')
parser.add_argument('validation_out', type=str,
                    help='path to store prediction on validation data')
parser.add_argument('metrics_out', type=str,
                    help='path to store training and testing metrics')
parser.add_argument('num_epoch', type=int,
                    help='number of training epochs')
parser.add_argument('hidden_units', type=int,
                    help='number of hidden units')
parser.add_argument('init_flag', type=int, choices=[1, 2],
                    help='weight initialization functions, 1: random')
parser.add_argument('learning_rate', type=float,
                    help='learning rate')


def args2data(args) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
str, str, str, int, int, int, float]:
    """
    DO NOT modify this function.

    Parse command line arguments, create train/test data and labels.
    :return:
    X_tr: train data *without label column and without bias folded in
        (numpy array)
    y_tr: train label (numpy array)
    X_te: test data *without label column and without bias folded in*
        (numpy array)
    y_te: test label (numpy array)
    out_tr: file for predicted output for train data (file)
    out_te: file for predicted output for test data (file)
    out_metrics: file for output for train and test error (file)
    n_epochs: number of train epochs
    n_hid: number of hidden units
    init_flag: weight initialize flag -- 1 means random, 2 means zero
    lr: learning rate
    """
    # Get data from arguments
    out_tr = args.train_out
    out_te = args.validation_out
    out_metrics = args.metrics_out
    n_epochs = args.num_epoch
    n_hid = args.hidden_units
    init_flag = args.init_flag
    lr = args.learning_rate

    X_tr = np.loadtxt(args.train_input, delimiter=',')
    y_tr = X_tr[:, 0].astype(int)
    X_tr = X_tr[:, 1:]  # cut off label column

    X_te = np.loadtxt(args.validation_input, delimiter=',')
    y_te = X_te[:, 0].astype(int)
    X_te = X_te[:, 1:]  # cut off label column

    return (X_tr, y_tr, X_te, y_te, out_tr, out_te, out_metrics,
            n_epochs, n_hid, init_flag, lr)


def shuffle(X, y, epoch):
    """
    DO NOT modify this function.

    Permute the training data for SGD.
    :param X: The original input data in the order of the file.
    :param y: The original labels in the order of the file.
    :param epoch: The epoch number (0-indexed).
    :return: Permuted X and y training data for the epoch.
    """
    np.random.seed(epoch)
    N = len(y)
    ordering = np.random.permutation(N)
    return X[ordering], y[ordering]


def zero_init(shape):
    """
    DO NOT modify this function.

    ZERO Initialization: All weights are initialized to 0.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    return np.zeros(shape=shape)


def random_init(shape):
    """

    RANDOM Initialization: The weights are initialized randomly from a uniform
        distribution from -0.1 to 0.1.

    :param shape: list or tuple of shapes
    :return: initialized weights
    """
    M, D = shape
    np.random.seed(M * D)  # Don't change this line!

    # TODO: create the random matrix here!
    # Hint: numpy might have some useful function for this
    # Hint: make sure you have the right distribution
    weights = np.random.uniform(low = -0.1, high=0.1, size=shape) # uniform distrbution here
    return weights


class SoftMaxCrossEntropy:

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """
        Implement softmax function.
        :param z: input logits of shape (num_classes,)
        :return: softmax output of shape (num_classes,)
        """

        ##
        # We will use cross entropy loss, l(y hat, y). If y represents our target output, which will be a one-hot
        # vector representing the correct class, and y hat represents the output of the network. 
        exp_vals = np.exp(z)
        return exp_vals/np.sum(exp_vals)

    def _cross_entropy(self, y: int, y_hat: np.ndarray) -> float:
        """
        Compute cross entropy loss.
        :param y: integer class label
        :param y_hat: prediction with shape (num_classes,)
        :return: cross entropy loss
        """
        # since y is int, need to convert it through one hot encoding 
        y_array = np.zeros_like(y_hat)
        y_array[y] = 1
        return -np.sum(y_array * np.log(y_hat))

    def forward(self, z: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Compute softmax and cross entropy loss.
        :param z: input logits of shape (num_classes,)
        :param y: integer class label
        :return:
            y: predictions from softmax as an np.ndarray
            loss: cross entropy loss
        """
        y_hat = self._softmax(z)
        loss = self._cross_entropy(y, y_hat)
        return y_hat, loss

    def backward(self, y: int, y_hat: np.ndarray) -> np.ndarray:
        """
        Compute gradient of loss w.r.t. ** softmax input **.
        Note that here instead of calculating the gradient w.r.t. the softmax
        probabilities, we are directly computing gradient w.r.t. the softmax
        input.

        Try deriving the gradient yourself (see Question 1.2(b) on the written),
        and you'll see why we want to calculate this in a single step.

        :param y: integer class label
        :param y_hat: predicted softmax probability with shape (num_classes,)
        :return: gradient with shape (num_classes,)
        """
        # from written part i get -y_k + \hat{y_k}
        # y is integer, y_hat is np.ndarray
        y_array = np.zeros_like(y_hat)
        y_array[y] = 1
        return -y_array + y_hat


class Sigmoid:
    def __init__(self):
        """
        Initialize state for sigmoid activation layer
        """
        self.cache = {} # create cache dictionary 

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Take sigmoid of input x.
        :param x: Input to activation function (i.e. output of the previous 
                  linear layer), with shape (output_size,)
        :return: Output of sigmoid activation function with shape
            (output_size,)
        """
        # TODO: perform forward pass and save any values you may need for
        #  the backward pass
        z_val = 1/(1+np.exp(-x))
        self.cache["z_val"] = z_val
        return z_val

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output of
            sigmoid activation
        :return: partial derivative of loss with respect to input of
            sigmoid activation
        """
        # calculated in written aprt that given partial derivative with respect output of sigmoid
        # the partial derivative with respect to the input of sigmoid is zj(1-zj). z is the sigmoid(aj)
        return dz * self.cache["z_val"] * (1 - self.cache["z_val"])


# This refers to a function type that takes in a tuple of 2 integers (row, col)
# and returns a numpy array (which should have the specified dimensions).
INIT_FN_TYPE = Callable[[Tuple[int, int]], np.ndarray]


class Linear:
    def __init__(self, input_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        :param input_size: number of units in the input of the layer 
                           *not including* the folded bias
        :param output_size: number of units in the output of the layer
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        # Initialize learning rate for SGD
        self.lr = learning_rate

        # TODO: Initialize weight matrix for this layer - since we are
        #  folding the bias into the weight matrix, be careful about the
        #  shape you pass in.
        #  To be consistent with the formulas you derived in the written and
        #  in order for the unit tests to work correctly,
        #  the first dimension should be the output size

        # first dim is output sie, second dim is num_input + 1 to account for folded intercept
        self.w = weight_init_fn((output_size, input_size+1))


        # TODO: set the bias terms to zero
        # bias is folded into weight matrix. first column should be 0
        self.w[:, 0] = 0

        # TODO: Initialize matrix to store gradient with respect to weights
        self.dw = np.zeros_like(self.w)

        # TODO: Initialize any additional values you may need to store for the
        #  backward pass here
        self.cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        :param x: Input to linear layer with shape (input_size,)
                  where input_size *does not include* the folded bias.
                  In other words, the input does not contain the bias column 
                  and you will need to add it in yourself in this method.
                  Since we train on 1 example at a time, batch_size should be 1
                  at training.
        :return: output z of linear layer with shape (output_size,)

        HINT: You may want to cache some of the values you compute in this
        function. Inspect your expressions for backprop to see which values
        should be cached.
        """
        # TODO: perform forward pass and save any values you may need for
        #  the backward pass
        x_with_bias = np.insert(x, 0, 1) # prepend x with 1
        # print("x with bias shape{}".format(x_with_bias.shape))
        # print("x shape{}".format(x.shape))
        result = np.dot(self.w, x_with_bias)
        self.cache['weights_wo_bias'] = self.w[:, 1:]
        self.cache["x_with_bias"] = x_with_bias
        #print("x_with_bias_shape is {}".format(x_with_bias.shape))
        return result

    def backward(self, dz: np.ndarray) -> np.ndarray:
        """
        :param dz: partial derivative of loss with respect to output z
            of linear
        :return: dx, partial derivative of loss with respect to input x
            of linear
        
        Note that this function should set self.dw
            (gradient of loss with respect to weights)
            but not directly modify self.w; NN.step() is responsible for
            updating the weights.

        HINT: You may want to use some of the values you previously cached in 
        your forward() method.
        """
        # TODO: implement
        dx = np.dot(self.cache['weights_wo_bias'].T, dz)
        # dw = partial l/ partial b * z^T
        # need to explicitly make x 2d to be able to transpose it since python
        # don't transpose 1d array. 
        temp = self.cache["x_with_bias"].reshape(-1,1)
        #print("x_with_bias trasposed shape is: {}".format(temp.shape))
        self.dw = np.dot(dz.reshape(-1,1), temp.T)

        return dx

    def step(self) -> None:
        """
        Apply SGD update to weights using self.dw, which should have been 
        set in NN.backward().
        """
        # TODO: implement
        #print(self.w)
        self.w = self.w - self.lr * self.dw



class NN:
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 weight_init_fn: INIT_FN_TYPE, learning_rate: float):
        """
        Initalize neural network (NN) class. Note that this class is composed
        of the layer objects (Linear, Sigmoid) defined above.

        :param input_size: number of units in input to network
        :param hidden_size: number of units in the hidden layer of the network
        :param output_size: number of units in output of the network - this
                            should be equal to the number of classes
        :param weight_init_fn: function that creates and initializes weight 
                               matrices for layer. This function takes in a 
                               tuple (row, col) and returns a matrix with 
                               shape row x col.
        :param learning_rate: learning rate for SGD training updates
        """
        self.weight_init_fn = weight_init_fn
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # TODO: initialize modules (see section 9.1.2 of the writeup)
        #  Hint: use the classes you've implemented above!
        self.linear1 = Linear(self.input_size, self.hidden_size, self.weight_init_fn, learning_rate)
        self.sigmoid = Sigmoid()
        self.linear2 = Linear(self.hidden_size, self.output_size, self.weight_init_fn, learning_rate)
        self.SoftMaxCrossEntropy = SoftMaxCrossEntropy()
    def forward(self, x: np.ndarray, y: int) -> Tuple[np.ndarray, float]:
        """
        Neural network forward computation. 
        Follow the pseudocode!
        :param x: input data point *without the bias folded in*
        :param y: prediction with shape (num_classes,)
        :return:
            y_hat: output prediction with shape (num_classes,). This should be
                a valid probability distribution over the classes.
            loss: the cross_entropy loss for a given example
        """
        # TODO: call forward pass for each layer
        a = self.linear1.forward(x)
        z = self.sigmoid.forward(a)
        b = self.linear2.forward(z)
        y_hat, entropy_loss = self.SoftMaxCrossEntropy.forward(b, y)

        return y_hat, entropy_loss

    def backward(self, y: int, y_hat: np.ndarray) -> None:
        """
        Neural network backward computation.
        Follow the pseudocode!
        :param y: label (a number or an array containing a single element)
        :param y_hat: prediction with shape (num_classes,)
        """
        # TODO: call backward pass for each layer
        gb = self.SoftMaxCrossEntropy.backward(y, y_hat)
        gz = self.linear2.backward(gb)
        ga = self.sigmoid.backward(gz)
        gx = self.linear1.backward(ga)

    def step(self):
        """
        Apply SGD update to weights.
        """
        # TODO: call step for each relevant layer
        self.linear1.step()
        self.linear2.step()

    def compute_loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute nn's average (cross entropy) loss over the dataset (X, y)
        :param X: Input dataset of shape (num_points, input_size)
        :param y: Input labels of shape (num_points,)
        :return: Mean cross entropy loss
        """
        # TODO: compute loss over the entire dataset
        #  Hint: reuse your forward function

        # haven't find a way to avoid using for-loop yet
        total_loss = 0
        for idx in range(y.shape[0]):
            _, current_loss = self.forward(X[idx,:], y[idx])
            total_loss = total_loss + current_loss
        return total_loss / y.shape[0]


    def train(self, X_tr: np.ndarray, y_tr: np.ndarray,
              X_test: np.ndarray, y_test: np.ndarray,
              n_epochs: int) -> Tuple[List[float], List[float]]:
        """
        Train the network using SGD for some epochs.
        :param X_tr: train data
        :param y_tr: train label
        :param X_test: train data
        :param y_test: train label
        :param n_epochs: number of epochs to train for
        :return:
            train_losses: Training losses *after* each training epoch
            test_losses: Test losses *after* each training epoch
        """
        # TODO: train network
        train_losses = [] # list of floats
        test_losses = [] # list of floats
        #print("total epochs: {}".format(n_epochs))
        for epoch in range(n_epochs):
            X_shu, y_shu = shuffle(X_tr, y_tr, epoch) # shuffle in every epoch
            for idx in range(0, y_shu.shape[0]):
                # compute forward
                y_hat, entropy_loss = self.forward(X_shu[idx, :], y_shu[idx])
                # Compute gradients via backprop:
                self.backward(y_shu[idx], y_hat)
                self.step()

            # evaluate train loss 
            mean_train_loss = self.compute_loss(X_shu, y_shu)
            train_losses.append(mean_train_loss)
            
            # evalute test loss after each epoch, don't use test data to update weights
            X_t_shu, y_t_shu = shuffle(X_test, y_test, epoch)
            mean_test_loss = self.compute_loss(X_t_shu, y_t_shu)
            test_losses.append(mean_test_loss)

        #print("train loss length: {}".format(len(train_losses)))
        return (train_losses, test_losses)

    def test(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute the label and error rate.
        :param X: input data
        :param y: label
        :return:
            labels: predicted labels
            error_rate: prediction error rate
        """
        # TODO: make predictions and compute error
        labels = []
        for idx in range(y.shape[0]):
            y_hat,_ = self.forward(X[idx, :], y[idx])
            label = np.argmax(y_hat)
            labels.append(label)
        error_rate = np.sum(labels!=y)/y.shape[0]
        return labels, error_rate



if __name__ == "__main__":
    args = parser.parse_args()
    # Note: You can access arguments like learning rate with args.learning_rate
    # Generally, you can access each argument using the name that was passed 
    # into parser.add_argument() above (see lines 24-44).

    # Define our labels
    labels = ["a", "e", "g", "i", "l", "n", "o", "r", "t", "u"]

    # Call args2data to get all data + argument values
    # See the docstring of `args2data` for an explanation of 
    # what is being returned.


    (X_tr, y_tr, X_test, y_test, out_tr, out_te, out_metrics,
     n_epochs, n_hid, init_flag, lr) = args2data(args)

    

    # for plotting
    import matplotlib.pyplot as plt

    n_epochs = 100
    n_hid = [5, 20, 50, 100, 200] # 50
    init_flag = 1 # random initializaton
    lr = 0.001
    plot_train_loss = []
    plot_val_loss = []



    for i in range(len(n_hid)):
        nn = NN(
            input_size=X_tr.shape[-1],
            hidden_size=n_hid[i],
            output_size=len(labels),
            weight_init_fn=zero_init if init_flag == 2 else random_init,
            learning_rate=lr
        )

        # train model
        # (this line of code is already written for you)
        train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)

        # test model and get predicted labels and errors 
        # (this line of code is written for you)



        #train_labels, train_error_rate = nn.test(X_tr, y_tr)
        plot_train_loss.append(train_losses[len(train_losses)-1])

        #test_labels, test_error_rate = nn.test(X_test, y_test)
        plot_val_loss.append(test_losses[len(test_losses)-1])

    plt.plot(n_hid, plot_train_loss, label='Training Loss', marker='o')
    plt.plot(n_hid, plot_val_loss, label='Validation Loss', marker='x')
    plt.title('Training and Validation Loss vs. Number of Hidden Units')
    plt.xlabel('Number of Hidden Units')
    plt.ylabel('Averaged Cross Entropy Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


    # nn = NN(
    #     input_size=X_tr.shape[-1],
    #     hidden_size=n_hid,
    #     output_size=len(labels),
    #     weight_init_fn=zero_init if init_flag == 2 else random_init,
    #     learning_rate=lr
    # )

    # # train model
    # # (this line of code is already written for you)
    # train_losses, test_losses = nn.train(X_tr, y_tr, X_test, y_test, n_epochs)

    # # test model and get predicted labels and errors 
    # # (this line of code is written for you)



    # train_labels, train_error_rate = nn.test(X_tr, y_tr)
    # #plot_train_loss.append(train_error_rate)

    # test_labels, test_error_rate = nn.test(X_test, y_test)
    #plot_val_loss.append(test_error_rate)

    # plt.plot(range(n_epochs), train_losses, label='Training Loss', marker='o')
    # plt.plot(range(n_epochs), test_losses, label='Validation Loss', marker='x')
    # plt.title('Training and Validation Loss vs. Number of Epochs')
    # plt.xlabel('Number of Epochs')
    # plt.ylabel('Averaged Cross Entropy Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.show()















    # Write predicted label and error into file
    # Note that this assumes train_losses and test_losses are lists of floats
    # containing the per-epoch loss values.
    # with open(out_tr, "w") as f:
    #     for label in train_labels:
    #         f.write(str(label) + "\n")
    # with open(out_te, "w") as f:
    #     for label in test_labels:
    #         f.write(str(label) + "\n")
    # with open(out_metrics, "w") as f:
    #     for i in range(len(train_losses)):
    #         cur_epoch = i + 1
    #         cur_tr_loss = train_losses[i]
    #         cur_te_loss = test_losses[i]
    #         f.write("epoch={} crossentropy(train): {}\n".format(
    #             cur_epoch, cur_tr_loss))
    #         f.write("epoch={} crossentropy(validation): {}\n".format(
    #             cur_epoch, cur_te_loss))
    #     f.write("error(train): {}\n".format(train_error_rate))
    #     f.write("error(validation): {}\n".format(test_error_rate))
