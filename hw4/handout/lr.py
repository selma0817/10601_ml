import numpy as np
import argparse



def dataloader(input_path):
    data_array = np.genfromtxt(input_path) #[num_sample, num_features]
    X = data_array[:, 1:]
    y = data_array[:, 0]
    return X, y


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int, 
    learning_rate : float,
    X_val : np.ndarray,
    y_val : np.ndarray
) -> None:
    train_loss = []
    val_loss = []

    for epoch in range(num_epoch):
        #print("Epoch: {}".format(epoch+1))
        neg_log_train = 0

        for i in range(X.shape[0]):
            # compute pointwise-gradient theta^T X
            cur_X = X[i].reshape(301, 1)
            w_X = np.dot(theta, cur_X)
            prob = sigmoid(w_X)
            grad = X[i] * (prob-y[i])
            theta = theta - learning_rate * grad
            # cal current training neg_loss 
            neg_log_train += y[i] * w_X - np.log(np.exp(w_X)+1)
        neg_log_train /= -X.shape[0]
        train_loss.append(neg_log_train)
        
        # cal validation loss
        neg_log_val = 0
        for i in range(X_val.shape[0]):
            cur_X_val = X_val[i].reshape(301, 1)
            w_X_val = np.dot(theta, cur_X_val)
            # cal current validation neg_loss 
            neg_log_val += y_val[i] * w_X_val - np.log(np.exp(w_X_val)+1)
        neg_log_val /= -X_val.shape[0]
        val_loss.append(neg_log_val)

    return theta, train_loss, val_loss


def plot_loss(train_loss, rate, val_loss = None):
    #plot the averaged_negative_log_likelihood
    import matplotlib.pyplot as plt
    if len(rate) != 0:
        for i in range(len(rate)):
            plt.plot(range(1, len(train_loss[i]) + 1), train_loss[i], label='$\eta = {}$'.format(rate[i]))
    #plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
        plt.xlabel("Epoch")
        plt.ylabel("Average Negative Log Likelihood")
        plt.title("Loss for Training using Different Learning Rate")
        plt.legend()
        plt.show()

def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    # i forgot to use sigmoid on the result of np.dot first time when i implement it 
    # that is why i kept getting 0.5 prediction result 
    results = sigmoid(np.dot(X, theta))
    predictions = (results >= 0.5).astype(int)
    return predictions

def compute_error(
    y_pred : np.ndarray, 
    y : np.ndarray
) -> float:
    false_pred = np.sum(y_pred != y)
    total = len(y)
    return false_pred / total


def write_error(train_err, test_err, out_path):
    with open(out_path,'w') as file:
        file.write("error(train): {:.6f}\n".format(train_err))
        file.write("error(test): {:.6f}".format(test_err))


def write_predictions(predictions, out_path):
    with open(out_path, 'w') as file:
        for p in predictions:
            file.write("{}\n".format(p))
        file.close()


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int, help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float, help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    num_epoch = args.num_epoch
    lr = args.learning_rate

    X_train, y_train = dataloader(args.train_input)
    X_val, y_val = dataloader(args.validation_input)
    X_test, y_test = dataloader(args.test_input)



    X_train = np.c_[(np.ones(X_train.shape[0]), X_train)]
    X_test = np.c_[(np.ones(X_test.shape[0]), X_test)]
    X_val = np.c_[(np.ones(X_val.shape[0]), X_val)]


    theta = np.zeros(X_train.shape[1])

    # train the theta

    #theta, avg_train_loss, avg_val_loss  = train(theta.T, X_train, y_train, num_epoch, lr, X_val, y_val)

    rate = [0.1, 0.01, 0.001]
    train_lr_loss = []
    theta_save = []
    val_lr_loss = []
    for i in range(len(rate)):
        theta_lr, avg_train_loss_lr, avg_val_loss_lr  = train(theta.T, X_train, y_train, num_epoch, rate[i], X_val, y_val)
        train_lr_loss.append(avg_train_loss_lr)
        theta_save.append(theta_lr)
        val_lr_loss.append(avg_val_loss_lr)
    plot_loss(train_lr_loss, rate)

    theta = theta_save[0]
    #prediction on training 
    train_preds = predict(theta.T, X_train)
    write_predictions(train_preds, args.train_out)
    train_err = compute_error(y_train, train_preds)


    #prediction on testing
    test_preds = predict(theta.T, X_test)
    write_predictions(test_preds, args.test_out)
    test_err = compute_error(y_test, test_preds)
    

    print(test_err)
    print(train_err)
    write_error(train_err, test_err, args.metrics_out)

    #plot_loss(avg_train_loss, avg_val_loss)


