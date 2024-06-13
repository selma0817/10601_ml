import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import argparse
from typing import List, Tuple
import time
import matplotlib.pyplot as plt
import numpy as np
from metrics import evaluate
from tqdm import tqdm

from torch.autograd import gradcheck

# Initialize the device type based on compute resources being used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DO NOT CHANGE THIS LINE OF CODE!!!!
torch.manual_seed(4)


class TextDataset(Dataset):
    def __init__(self, train_input: str, word_to_idx: dict, tag_to_idx: dict, idx_to_tag: dict):
        """
        Initialize the dictionaries, sequences, and labels for the dataset

        :param train_input: file name containing sentences and their labels
        :param word_to_idx: dictionary which maps words (str) to indices (int). Should be initialized to {} 
            outside this class so that it can be reused for test data. Will be filled in by this class when training.
        :param tag_to_idx: dictionary which maps tags (str) to indices (int). Should be initialized to {} 
            outside this class so that it can be reused for test data. Will be filled in by this class when training.
        :param idx_to_tag: Inverse dictionary of tag_to_idx, which maps indices (int) to tags (str). Should be initialized to {} 
            outside this class so that it can be reused when evaluating the F1 score of the predictions later on. 
            Will be filled in by this class when training.
        """
        self.sequences = []
        self.labels = []
        self.num_seq = 0
        # self.unique_words = 0
        # self.unique_tags = 0
        i = 0 # index counter for word dict
        j = 0 # index counter for tag dict

        # for all sequences, convert the words/labels to indices using 2 dicts,
        # append these indices to the 2 lists, and add the resulting lists of
        # word/label indices to the accumulated dataset

        # open the file and load all contents 
        with open(train_input, 'r') as f:
            # strip() and split() may be useful to you here
            # note we have to strip() else in block.split('\t') will be 
            # one last term ("") only have 1 item inside.
            content = f.read().strip()
            # seperate each sequence by \n\n
            blocks = content.split('\n\n')
            #print("# of sequences : {}".format(len(blocks)))
            self.num_seq = len(blocks)
            for block in blocks:
                sequence_idx = []
                tag_idx = []
                tokens = block.split('\n')
                for token in tokens:
                    word, tag = token.split('\t')
                    # create a mapping of unique words
                    if word not in word_to_idx:
                        word_to_idx[word] = i
                        i += 1
                    if tag not in tag_to_idx:
                        tag_to_idx[tag] = j
                        idx_to_tag[j] = tag
                        j += 1
                    sequence_idx.append(word_to_idx[word])
                    tag_idx.append(tag_to_idx[tag])
                self.sequences.append(sequence_idx)
                self.labels.append(tag_idx)
            # self.unique_words = i
            # self.unique_tags = j
    def __len__(self):
        """
        :return: Length of the text dataset (# of sentences)
        """
        # you must make sure to return the length of the dataset (the total number of sequences) in the len
        # function. length = num of sequence
        return self.num_seq

    def __getitem__(self, idx):
        """
        Return the sequence of words and corresponding labels given input index

        :param idx: integer of the index to access
        :return word_tensor: sequence of words as a tensor
        :return label_tensor: sequence of labels as a tensor
        """
        word_tensor = torch.tensor(self.sequences[idx])
        label_tensor = torch.tensor(self.labels[idx])
        return word_tensor, label_tensor


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input: nn.Parameter, weight: nn.Parameter, bias: nn.Parameter):
        """
        Manual implementation of a Layer Linear forward computation that 
        also caches parameters for the backward computation. 

        :param ctx: context object to store parameters
        :param input: training example tensor of shape (batch_size, in_features)
        :param weight: weight tensor of shape (out_features, in_features)
        :param bias: bias tensor of shape (out_features)
        :return: forward computation output of shape (batch_size, out_features)
        """
        ctx.save_for_backward(input, weight)
        # suppose dim(in_features) = n
        # input shape is: (1, n)
        # suppose dim(out_features) = k
        # weight shape is: (k, n)
        # (1, n) matmul (n, k)
        # bias shape is: (k,)
        # after reshape bias is (1, k)

        output = (torch.matmul(input, torch.transpose(weight, 0, 1)) 
                                            + bias.reshape((1, weight.shape[0])))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Manual implementation of a Layer Linear backward computation 
        using the cached parameters from forward computation

        :param ctx: context object to access stored parameters
        :param grad_output: partial derviative w.r.t Linear outputs (What is the shape?)
        :returns:
            g_input: partial derivative w.r.t Linear inputs (Same shape as inputs)
            g_weight: partial derivative w.r.t Linear weights (Same shape as weights)
            g_bias: partial derivative w.r.t Linear bias (Same shape as bias, remember that bias is 1-D!!!)
        """
        input, weight = ctx.saved_tensors
        g_input = torch.matmul(grad_output, weight)
        g_weight = torch.matmul(torch.transpose(grad_output, 0, 1), input)
        g_bias = torch.sum(grad_output, dim=0)

        return g_input, g_weight, g_bias
    

class TanhFunction(Function):
    @staticmethod
    def forward(ctx, input):
        """
        Take the Tanh of input parameter

        :param ctx: context object to store parameters
        :param input: Activiation input (output of previous layers)
        :return: output of tanh activation of shape identical to input
        """
        # can use torch.tanh in the function
        
        output = torch.tanh(input)
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs backward computation of Tanh activation

        :param ctx: context object to access stored parameters
        :param grad_output: partial deriviative of loss w.r.t Tanh outputs
        :return: partial deriviative of loss w.r.t Tanh inputs
        """
        # partial deriviative of loss w.r.t Tanh inputs
        # is 1-tanh^2(x)
        output, = ctx.saved_tensors
        return grad_output * (1 - output**2)



class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        """
        Takes the ReLU of input parameter

        :param ctx: context object to store parameters
        :param input: Activation input (output of previous layers) 
        :return: Output of ReLU activation with shape identical to input
        """
        ctx.save_for_backward(input)
        # only keep term >= 0
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs backward computation of ReLU activation

        :param ctx: context object to access stored parameters
        :param grad_output: partial deriviative of loss w.r.t ReLU output
        :return: partial deriviative of loss w.r.t ReLU input
        """
        input, = ctx.saved_tensors
        mask = (input > 0).float()
        # > 0 term have drivative of 1, <=0 term has 0
        return grad_output * mask


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Initialize the dimensions and the weight and bias matrix for the linear layer.

        :param in_features: units in the input of the layer
        :param out_features: units in the output of the layer
        """

        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # See https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        bound = torch.sqrt(1 / torch.tensor([in_features])).item()

        self.weight = nn.Parameter(nn.init.uniform_(
            torch.empty(out_features, in_features), a=-1*bound, b=bound))
        self.bias = nn.Parameter(nn.init.uniform_(
            torch.empty(out_features), a=-1*bound, b=bound))

    def forward(self, x):
        """
        Wrapper forward method to call the self-made Linear layer

        :param x: Input into the Linear layer, of shape (batch_size, in_features)
        """
        return LinearFunction.apply(x, self.weight, self.bias)


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        """
        Wrapper forward method to call the Tanh activation layer

        :param x: Input into the Tanh activation layer
        :return: Output of the Tanh activation layer
        """
        return TanhFunction.apply(x)


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """
        Wrapper forward method to call the ReLU activation layer

        :param x: Input into the ReLU activation layer
        :return: Output of the ReLU activation layer
        """
        return ReLUFunction.apply(x)


class RNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, activation: str):
        """
        Initialize the embedding dimensions, hidden layer dimensions, 
        hidden Linear layers, and activation.

        :param embedding_dim: integer of the embedding size
        :param hidden_dim: integer of the dimension of hidden layer 
        :param activation: string of the activation type to use (Tanh, ReLU)
        """
        super(RNN, self).__init__()
        # The init method sets all the parameters for the model. This includes 
        # embedding dim is input size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # from lecture we have  h_t = activate(W_xh * xt + W_hh * h_t-1 + b_h)
        # W_xh is input to hidden; W_hh is hidden to hidden

        self.W_xh = Linear(self.embedding_dim, self.hidden_dim)
        self.W_hh = Linear(self.hidden_dim, self.hidden_dim)
        if (activation == 'relu'):
            self.activation = ReLU()
        elif (activation == 'tanh'):
            self.activation = Tanh()
        else:
            print("choose between ReLU and TanH")
    
    def forward(self, embeds):
        """
        Computes the forward pass for the RNN using the hidden layers
        and the input represented by embeddings. Sets initial hidden state to zeros.

        :param embeds: a batch of training examples converted to embeddings of size (batch_size, seq_length, embedding_dim)
        :returns: 
            outputs: list containing the final hidden states at each sequence length step. Each element has size (batch_size, hidden_dim),
            and has length equal to the sequence length
        """
        #embeds = embeds.squeeze(1)
        #print("embeds shape is: {}".format(embeds.shape))

        (batch_size, seq_length, _) = embeds.shape
        #output = [] # each element has size (batch_size, hidden_dim)

        hidden = torch.zeros(batch_size, self.hidden_dim)
        outputs = []
        for t in range(seq_length):
            x_t = embeds[:, t, :]
            hidden = self.activation(self.W_xh.forward(x_t) + self.W_hh.forward(hidden))
            outputs.append(hidden) # each hidden has size : (batch_size, hidden_dim)
        return outputs # sine hidden is [1, 32] the output is [[1, 32]]


class TaggingModel(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int, 
                hidden_dim: int, activation: str):
        """
        Initialize the underlying sequence model, activation name, 
        sequence embeddings and linear layer for use in the forward computation.
        
        :param vocab_size: integer of the number of unique "words" in our vocabulary
        :param tagset_size: integer of the the number of possible tags/labels (desired output size)
        :param embedding_dim: integer of the size of our sequence embeddings
        :param hidden_dim: integer of the hidden dimension to use in the Linear layer
        :param activation: string of the activation name to use in the sequence model
        """
        # 1. initialize the embedding layer (Hint: Use nn.Embedding), 
        # 2. the RNN class,
        # 3. the final Linear layer for use on the outputs of the RNN.

        super(TaggingModel, self).__init__()

        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        self.RNN = RNN(embedding_dim, hidden_dim, activation)
        self.W_hy = Linear(hidden_dim, tagset_size)


    def forward(self, sentences):
        """
        Perform the forward computation of the model (prediction), given batched input sentences.

        :param sentences: batched string sentences of shape (batch_size, seq_length) to be converted to embeddings 
        :return tag_distribution: concatenated results from the hidden to out layers (batch_size, seq_len, tagset_size)
        """
        # During the forward pass, the embedding layer takes indices as input 
        # (e.g., word indices in a vocabulary) and returns the corresponding embeddings
        #print("sentence shape is {}".format(sentences.shape)) #sentences shaped as [batch_size, seq_length]
        embeds = self.embedding_layer(sentences)
        #print("embeds shape: {}".format(embeds.shape)) # [batch_size, seq_length, embedding_dim]

        # The output from the RNN is a list. I would suggest iterating through that list and applying the Linear Module 
        # to each tensor in that list, and stacking those outputs together to create a new tensor. Trying to convert the 
        # list to a tensor and then feeding that tensor into the Linear Module may cause shape issues.
        tag_outputs = []

        hidden_list = self.RNN(embeds) # output is a list with each element having dim = (batch_size, hidden_dim)
        for ht in hidden_list:
            #print("ht has shape: {}".format(ht.shape))
            tag_t = self.W_hy(ht)  # has shape (batch_size, target_size)
            #print("tag_t shape: {}".format(tag_t.shape))
            tag_outputs.append(tag_t)
        tag_dist = torch.stack(tag_outputs, dim=1)
        #print("tag_dist shape: {}".format(tag_dist.shape))
        #print(tag_dist)
        return tag_dist

        
    

def calc_metrics(true_list, pred_list, tags_dict):
    """
    Calculates precision, recall and f1_score for lists of tags
    You aren't required to implement this function, but it may be helpful
    in modularizing your code.

    :param true_list: list of true/gold standard tags, in index form
    :param pred_list: list of predicted tags, in index form
    :param tags_dict: dictionary of indices to tags
    :return:
        (optional) precision: float of the overall precision of the two lists
        (optional) recall: float of the overall recall of the two lists
        f1_score: float of the overall f1 score of the two lists
    """
    true_list_tags = [tags_dict[i] for i in true_list]
    pred_list_tags = [tags_dict[i] for i in pred_list]
    precision, recall, f1_score = evaluate(true_list_tags, pred_list_tags)
    return precision, recall, f1_score


def train_one_epoch(model, dataloader, loss_fn, optimizer):
    """
    Trains the model for exactly one epoch using the supplied optimizer and loss function

    :param model: model to train 
    :param dataloader: contains (sentences, tags) pairs
    :param loss_fn: loss function to call based on predicted tags (tag_dist) and true label (tags)
    :param optimizer: optimizer to call after loss calculated
    """
    # running_loss = 0
    # last_loss = 0
    model.train()
    for i, data in enumerate(dataloader):
        sentences, tags = data
        optimizer.zero_grad()
        tag_dist = model(sentences) # tag_dist has shape [(batch_size,seq_length, target_size)]
        # (batch_size * seq_length, num_classes)
        # (batch_size * seq_length, )
        loss = loss_fn(tag_dist.reshape(-1,tag_dist.shape[-1]), tags.reshape(-1))
        loss.backward()
        optimizer.step()

    

def predict_and_evaluate(model, dataloader, tags_dict, loss_fn):
    """
    Predicts the tags for the input dataset and calculates the loss, accuracy, and f1 score

    :param model: model to use for prediction
    :param dataloader: contains (sentences, tags) pairs
    :param tags_dict: dictionary of indices to tags
    :param loss_fn: loss function to call based on predicted tags (tag_dist) and true label (tags)
    :return:
        loss: float of the average loss over dataset throughout the epoch
        accuracy: float of the average accuracy over dataset throughout the epoch
        f1_score: float of the overall f1 score of the dataset
        all_preds: list of all predicted tag indices
    """
    total_loss = 0
    f1_score = 0
    all_preds = []
    all_true = []
    with torch.no_grad():
        model.eval()
        for inputs, labels in dataloader: # e.g. sentence is tensor([1, 2, 3, 5, 4]), tag is tensor([4, 5, 7, 2, 4])
            outputs = model(inputs) #[batch_size, seq_len, num_classes]
            loss = loss_fn(outputs.reshape(-1,outputs.shape[-1]), labels.reshape(-1))
            total_loss += loss.item()

            # get predicted tag by max
            pred_tag_idx = torch.argmax(outputs, dim=2) # tag_dist is (batch_size, seq_length, num_classes)
            all_preds.extend(pred_tag_idx.flatten().tolist())
            all_true.extend(labels.flatten().tolist())

        average_loss = total_loss/len(dataloader)
        # calc_metric 
        average_accuracy = np.mean(np.array(all_preds) == np.array(all_true))
        _,_,f1_score = calc_metrics(all_true, all_preds, tags_dict)


        return average_loss, average_accuracy, f1_score, all_preds



def train(train_dataloader, test_dataloader, model, optimizer, loss_fn, 
            tags_dict, num_epochs: int):
    """
    Trains the model for the supplied number of epochs. Performs evaluation on 
    test dataset after each epoch and accumulates all train/test accuracy/losses.

    :param train_dataloader: contains training data
    :param test_dataloader: contains testing data
    :param model: model module to train
    :param optimizer: optimizer to use in training loop
    :param loss_fn: loss function to use in training loop
    :param tags_dict: dictionary of indices to tags
    :param num_epochs: number of epochs to train
    :return:
        train_losses: list of integers (train loss across epochs)
        train_accuracies: list of integers (train accuracy across epochs)
        train_f1s: list of integers (train f1 score across epochs)        
        test_losses: list of integers (test loss across epochs)
        test_accuracies: list of integers (test accuracy across epochs)
        test_f1s: list of integers (test f1 score across epochs)
        final_train_preds: list of tags (final train predictions on last epoch)
        final_test_preds: list of tags (final test predictions on last epoch)
    """
    train_loss = []
    train_accuracy = []
    train_f1s = []
    
    test_loss = []
    test_accuracy = []
    test_f1s = []


    for epoch in range(num_epochs):
        print("epoch: {}".format(1+epoch))
        train_one_epoch(model, train_dataloader, loss_fn, optimizer)
        ave_loss, ave_accuracy, epoch_f1_score, all_preds_train = predict_and_evaluate(model, train_dataloader, tags_dict, loss_fn)
        print(ave_loss)
        train_loss.append(ave_loss)
        train_accuracy.append(ave_accuracy)
        train_f1s.append(epoch_f1_score)

        with torch.no_grad():
            ave_te_loss, ave_te_accuracy, test_epoch_f1_score, all_preds_test = predict_and_evaluate(model, test_dataloader, tags_dict, loss_fn)
            test_loss.append(ave_te_loss)
            test_accuracy.append(ave_te_accuracy)
            test_f1s.append(test_epoch_f1_score)


    return train_loss, train_accuracy, train_f1s, test_loss, test_accuracy, test_f1s,  all_preds_train, all_preds_test

def main(train_input: str, test_input: str, embedding_dim: int, 
         hidden_dim: int,  num_epochs: int, activation: str):
    """
    Main function that creates dataset/dataloader, initializes the model, optimizer, and loss.
    Also calls training and inferences loops.
    
    :param train_input: string of the training .txt file to read
    :param test_input: string of the testing .txt file to read
    :param embedding_dim: dimension of the input embedding vectors
    :param hidden_dim: dimension of the hidden layer of the model
    :param num_epochs: number of epochs for the training loop
    :param activation: string of the type of activation to use in seq model

    :return: 
        train_losses: train loss from the training loop
        train_accuracies: train accuracy from the training loop
        train_f1s: train f1 score from the training loop
        test_losses: test loss from the training loop
        test_accuracies: test accuracy from the training loop
        test_f1s: test f1 score from the training loop
        train_predictions: final predicted labels from the train dataset
        test_predictions: final predicted labels from the test dataset
    """
    # 1. creates dataset/dataloader
    # train loader 
    word_to_idx = {}
    tag_to_idx = {}
    idx_to_tag = {}
    
    
    train_dataset = TextDataset(train_input, word_to_idx, tag_to_idx, idx_to_tag) #  shuffle=False
    test_dataset = TextDataset(test_input, word_to_idx, tag_to_idx, idx_to_tag) # shuffle=False

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # 2. initializes the model
    #vocab_size =  train_dataset.unique_words
    #tagset_size = train_dataset.unique_tags # is it because of making vocab size only unique of train?
    #print("vocab_size: {}".format(vocab_size))
    #print("tagset_size: {}".format(tagset_size))

    vocab_size = len(word_to_idx)
    tagset_size = len(tag_to_idx)


    model = TaggingModel(vocab_size, tagset_size, embedding_dim, hidden_dim, activation)
    #self, vocab_size: int, tagset_size: int, embedding_dim: int, hidden_dim: int, activation: str
    # Please use the Adam optimizer with the default learning rate (0.001) when training to match the 
    # metrics in the writeup and to pass the autograder.
    lr = 0.001
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    

    train_loss, train_accuracy, train_f1s, test_loss, test_accuracy, test_f1s,  all_preds_train, all_preds_test  = train(train_dataloader, test_dataloader, model, optim, loss_fn, 
            idx_to_tag, num_epochs)

    return train_loss, train_accuracy, train_f1s, test_loss, test_accuracy, test_f1s,  all_preds_train, all_preds_test


if __name__ == '__main__':
    # DO NOT MODIFY THIS ARGPARSE CODE
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    parser = argparse.ArgumentParser()
    # parser.add_argument('--embedding_dim', type=int, help='size of the embedding vector')
    # parser.add_argument('--hidden_dim', type=int, help='size of the hidden layer')
    # parser.add_argument('--num_epochs', type=int, help='number of epochs')
    # parser.add_argument('--activation', type=str, choices=["tanh", "relu"], help='activation layer to use')
    # parser.add_argument('train_input', type=str, help='path to training input .txt file')
    # parser.add_argument('test_input', type=str, help='path to testing input .txt file')
    # parser.add_argument('train_out', type=str, help='path to .txt file to write training predictions to')
    # parser.add_argument('test_out', type=str, help='path to .txt file to write testing predictions to')
    # parser.add_argument('metrics_out', type=str, help='path to .txt file to write metrics to')
        
    # args = parser.parse_args()
    #Call the main function
    train_input = "./data/en.train_10.twocol.oov"
    test_input = "./data/en.val_10.twocol.oov"
    embedding_dims = [50]
    hidden_dim = 50
    num_epochs = 10
    activation = "relu"
    # 4 total line showing f1 score

    train_losses, train_accuracies, train_f1s, test_losses, test_accuracies, test_f1s, train_predictions, test_predictions = main(train_input, test_input, embedding_dim, embedding_dim, num_epochs, activation)

    # import matplotlib.pyplot as plt
    # epochs = range(1, len(all_f1[0]) + 1)  # Assuming all lists are the same length
    # # Plotting
    # plt.figure(figsize=(10, 6))  # Optional: Adjusts the size of your graph

    # plt.plot(epochs, all_f1[0], label='Train (64)', marker='v')
    # plt.plot(epochs, all_f1[1], label='Val (64)', marker='v')
    # plt.plot(epochs, all_f1[2], label='Train (128)', marker='o')
    # plt.plot(epochs, all_f1[3], label='Val (128))', marker='o')
    # plt.plot(epochs, all_f1[4], label='Train (512)', marker='*')
    # plt.plot(epochs, all_f1[5], label='Val (512)', marker='*')

    # plt.title('F1 Score per Epoch')
    # plt.xlabel('Epoch')
    # plt.ylabel('F1 Score')
    # plt.legend()
    # plt.grid(True)  # Optional: Adds a grid for easier visualization
    # # Save the plot as a PNG file
    # plt.savefig('different_dim.png')
    # plt.show()

    


    with open(args.train_out, 'w') as f:
        for pred in train_predictions:
            f.write(str(int(pred)) + '\n')

    with open(args.test_out, 'w') as f:
        for pred in test_predictions:
            f.write(str(int(pred)) + '\n')

    train_acc_out = train_accuracies[-1]
    train_f1_out = train_f1s[-1]
    test_acc_out = test_accuracies[-1]
    test_f1_out = test_f1s[-1]

    with open(args.metrics_out, 'w') as f:
        f.write('accuracy(train): ' + str(round(train_acc_out, 6)) + '\n')
        f.write('accuracy(test): ' + str(round(test_acc_out, 6)) + '\n')
        f.write('f1(train): ' + str(round(train_f1_out, 6)) + '\n')
        f.write('f1(test): ' + str(round(test_f1_out, 6)))



    # word_to_idx, tag_to_idx, idx_to_tag = 
    #data = TextDataset(train_input, word_to_idx, tag_to_idx, idx_to_tag)

    # # generate random input: 
    # input = torch.randn((1, 5), dtype=torch.double, requires_grad=True)

    # # grade check expect tuple for input
    # tanh_test_passed = gradcheck(TanhFunction.apply, (input,), eps=1e-6, atol=1e-5, rtol=1e-3)
    # print(f"Gradcheck passed tanh: {tanh_test_passed}")

    # relu_test_passed = gradcheck(ReLUFunction.apply, (input,), eps=1e-6, atol=1e-5, rtol=1e-3)
    # print(f"Gradcheck passed relu: {relu_test_passed}")

    #     # suppose dim(in_features) = n
    #     # input shape is: (1, n)
    #     # suppose dim(out_features) = k
    #     # weight shape is: (k, n)
    #     # (1, n) matmul (n, k)
    #     # bias shape is: (k,)
    #     # after reshape bias is (1, k)

    # weight = torch.randn((3, 5), dtype=torch.double, requires_grad=True)
    # bias = torch.randn(3, dtype=torch.double, requires_grad=True)
    # # linear_test_passed = gradcheck(LinearFunction.apply, (input,weight, bias), eps=1e-6, atol=1e-5, rtol=1e-3)
    # # print(f"Gradcheck passed linear: {linear_test_passed}")



    