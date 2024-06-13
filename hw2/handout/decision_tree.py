import argparse
import numpy as np
from inspection import data_factory, cal_entropy, majority_vote_classifier, write_inspection
import matplotlib.pyplot as plt

def get_mutual_info(data, attrs, cur_attr): ## for example, I(Y; A) = H(Y) - H(Y|A)
    index = np.argmax(attrs == cur_attr) # need to add 1 since data include label col but attrs don't
    total_entropy = cal_entropy(data) 
    cur_col = data[:, index]
    mask_0 = (cur_col == 0).flatten()
    mask_1 = (cur_col == 1).flatten()
    mutual_info = total_entropy - conditional_entropy(data, cur_col, mask_0, mask_1)
    return mutual_info


def conditional_entropy(data, cur_col, mask_0, mask_1):
    size = len(data[:, -1])
    data_0 = data[mask_0, :]
    data_1 = data[mask_1, :]
    prob_0 = len(data_0)/size
    prob_1 = len(data_1)/size

    return prob_0 * cal_entropy(data_0) + prob_1 * cal_entropy(data_1)

class decision_tree:
    def __init__(self, max_depth):
        self.max_depth = max_depth
        self.base_classifier = majority_vote_classifier()
        self.root_node = None
        self.max_idx = None
        self.max_attr = None
        self.num_attrs = None
    def build_tree(self, data, attrs, depth):
        node = Node()
        #print(data.shape)
        if depth==0:
            node.original_num0 = np.sum(data[:,-1]==0)
            node.original_num1 = np.sum(data[:,-1]!=0)
        # base case
        self.num_attrs = data.shape[1]-1
        if(depth >= self.max_depth or (depth>=self.num_attrs) or
        cal_entropy(data)==0):
            #or len(np.unique(data[:,-1]))==1
            self.base_classifier.fit(data)
            node.vote = self.base_classifier.predict()
        else:
            max_mutual_info = 0 
            for cur_idx, cur_attr in enumerate(attrs):
                #if cur_attr not in self.visited_attrs:
                temp = get_mutual_info(data, attrs, cur_attr)
                if temp > max_mutual_info:
                    max_mutual_info = temp
                    self.max_attr = cur_attr
                    self.max_idx = cur_idx
            # max_header correspond to col with largest mutual info
                node.attr = self.max_attr
            #self.visited_attrs.add(self.max_attr)
                

            right_mask = data[:, self.max_idx] == 0
            r_data = data[right_mask]
            left_mask = data[:, self.max_idx] == 1
            l_data = data[left_mask]
            print(self.max_idx)
            node.zero_branch_dist = [np.sum(r_data[:,-1]==0), np.sum(r_data[:,-1]==1)]
            node.one_branch_dist = [np.sum(l_data[:,-1]==0), np.sum(l_data[:,-1]==1)]

            node.right = self.build_tree(data[right_mask,:], attrs, depth+1)

            node.left = self.build_tree(data[left_mask,:], attrs, depth+1)

        return node ## return the root node
    
    def fit(self, data, attrs):
        self.root_node = self.build_tree(data, attrs, depth=0)
        return self.root_node
    
    def predict_single(self, node, attrs_dict):

        if node is not None:
            #print('not None')
            if node.attr in attrs_dict.keys(): #not leaf node yet
                val = attrs_dict[node.attr]
                if val==0: # because 0 branch on right side
                    return self.predict_single(node.right, attrs_dict) 
                else:
                    return self.predict_single(node.left, attrs_dict)  
            else: # reach leaf node
                return node.vote
        else:
            return None

    def make_attrs_dict(self, single_input, headers):
        attrs_dict = {}
        for idx, header in enumerate(headers):
            attrs_dict[header] = single_input[idx]
        return attrs_dict

    def predict(self, node, test_data, headers):
        predictions = []
        for single_input in test_data:
            dict = self.make_attrs_dict(single_input, headers)
            pred = self.predict_single(node, dict)
            predictions.append(pred)
        return predictions


    def print_tree(self, Node, f, depth=0):
    # reach leaf node
        if Node is not None:
            if Node.attr is not None:
                if depth==0:
                    print(f"[{Node.original_num0} 0/{Node.original_num1} 1]",file=f)
                print("| " * (depth+1), end="",file=f)
                print(f"{Node.attr} = 0: [{Node.zero_branch_dist[0]} 0/{Node.zero_branch_dist[1]} 1]",file=f)
                self.print_tree(Node.right, f, depth + 1)
                print("| " * (depth+1), end="",file=f)
                print(f"{Node.attr} = 1: [{Node.one_branch_dist[0]} 0/{Node.one_branch_dist[1]} 1]",file=f)
                self.print_tree(Node.left, f, depth + 1)


class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self):
        self.left = None
        self.right = None
        self.attr = None # attr is what this node is splitting on
        self.vote = None # only leave node have vote
        self.zero_branch_dist = None
        self.one_branch_dist = None
        self.original_num0 = None
        self.original_num1 = None

        # maybe add information of 1 and 0 count in node

def write_to_output(predictions, output_file_path):
    with open(output_file_path, 'w') as file:
        for pred in predictions:
            file.write(f"{pred}\n")

def write_metric(train_predictions, test_predictions, train_labels, test_labels, metric_path):

    train_error = np.sum(train_labels != np.array(train_predictions))/len(train_labels)
    test_error = np.sum(test_labels != np.array(test_predictions))/len(test_labels)
    print(train_error)
    print(test_error)
    print('----')
    with open(metric_path, 'w') as file:
        file.write(f"error(train): {train_error }\n")
        file.write(f"error(test): {test_error }\n")

def pipeline(train_input, test_input, max_depth, train_out, test_out, metrics_out, print_out):
    
    d_factory = data_factory()
    tree = decision_tree(max_depth)
    
    
    # train data 
    train_data = d_factory.get_data(args.train_input) # all data including y
    train_attrs = d_factory.get_attrs(args.train_input)#[:-1] # all headers except one for label
    train_labels = d_factory.get_labels(args.train_input)
    
    # test data
    test_data = d_factory.get_data(args.test_input) # all data including y
    test_attrs = d_factory.get_attrs(args.test_input)#[:-1] # all headers except one for label
    test_labels = d_factory.get_labels(args.test_input)

    r_node = tree.fit(train_data, train_attrs)

    train_predictions = tree.predict(r_node, train_data, train_attrs)
    test_predictions = tree.predict(r_node, test_data, test_attrs)

    # write outputs
    write_to_output(train_predictions, train_out)
    write_to_output(test_predictions, test_out)

    # write metrics
    write_metric(train_predictions, test_predictions, train_labels, test_labels, args.metrics_out)

    # print tree
    with open(args.print_out, "w") as f:
        tree.print_tree(r_node, f, 0)


def plot_errors(train_input, test_input):
    d_factory = data_factory()

    # train data 
    train_data = d_factory.get_data(args.train_input) # all data including y
    train_attrs = d_factory.get_attrs(args.train_input)#[:-1] # all headers except one for label
    train_labels = d_factory.get_labels(args.train_input)
    
    # test data
    test_data = d_factory.get_data(args.test_input) # all data including y
    test_attrs = d_factory.get_attrs(args.test_input)#[:-1] # all headers except one for label
    test_labels = d_factory.get_labels(args.test_input)


    # Initialize lists to store training and testing errors
    train_errors = []
    test_errors = []
    
    for depth in range(0, len(train_attrs) + 1):
        tree = decision_tree(depth) 
        r_node = tree.fit(train_data, train_attrs)

        train_predictions = tree.predict(r_node, train_data, train_attrs)
        test_predictions = tree.predict(r_node, test_data, test_attrs)

        train_error = np.sum(train_labels != np.array(train_predictions))/len(train_labels)
        test_error = np.sum(test_labels != np.array(test_predictions))/len(test_labels)

        train_errors.append(train_error)
        test_errors.append(test_error)

    # Plotting the errors
    plt.plot(range(0, len(train_attrs) + 1), train_errors, label='Training Error')
    plt.plot(range(0, len(train_attrs) + 1), test_errors, label='Testing Error')

    # Labeling the axes and title
    plt.xlabel('Depth of the Tree')
    plt.ylabel('Error')
    plt.title('Training and Testing Errors vs. Tree Depth')
    # Adding a legend
    plt.legend()
    # Save the plot to a file
    plt.savefig('heart.png')
    # Show the plot
    plt.show()

if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()
    

    #pipeline(args.train_input, args.test_input, args.max_depth, args.train_out, args.test_out, args.metrics_out, args.print_out)
    plot_errors(args.train_input, args.test_input)



