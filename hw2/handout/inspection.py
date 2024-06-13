import sys
import numpy as np
import argparse

class data_factory:
    def __init__(self)-> None:
        self.data = None
        
        self.attrs = None
        self.labels = None

    def get_data(self, tsv_file_path):
        self.data = np.genfromtxt(tsv_file_path, delimiter='\t', skip_header=True)
        return self.data
    def get_attrs(self, tsv_file_path):
        temp = np.genfromtxt(tsv_file_path, delimiter='\t', dtype=str, max_rows=1)
        self.attrs = temp[:-1] # all but last 
        return self.attrs
    def get_labels(self, tsv_file_path):
        self.labels = self.data[:,-1]
        return self.labels


def cal_entropy(data):
    labels = data[:,-1]
    #num_labels = len(labels)
    values, counts = np.unique(labels, return_counts=True)
    probs = counts/len(data)
    entropy = -np.sum(probs*np.log2(probs))
    return entropy

class majority_vote_classifier:
    def __init__(self)-> None:
        self.majority = None
    
    def fit(self, data):
        label = data[:,-1]
        self.majority = int(np.sum(label==1)>=np.sum(label==0))
    
    def predict(self):
        return self.majority



def write_inspection(data, entropy, classifier, output_filename):
    with open(output_filename, 'w') as file:
        true_label = data[:,-1]
        pred_label = classifier.predict()         
        error = np.sum(true_label != (np.full(len(true_label), pred_label)))/len(true_label)
        file.write(f"entropy: {entropy:.4f}\n")
        file.write(f"error: {error:.4f}\n")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=str, help='path to input .tsv file')
    parser.add_argument("output", type=str, help='path to output .txt file')
    args = parser.parse_args()

    input_filename = args.input
    output_filename = args.output

    # convert from tsv to np
    d_factory = data_factory()
    
    np_data = d_factory.get_data(input_filename)
    entropy = cal_entropy(np_data)

    # train classifier
    classifier = majority_vote_classifier()
    classifier.fit(np_data)

    write_inspection(np_data, entropy, classifier, output_filename)


