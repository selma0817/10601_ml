
import sys
import numpy as np


def tsv_to_numpy(tsv_file_path):
    # np.genfromtxt is more concise compare to 
    # readlines from file method
    return np.genfromtxt(tsv_file_path, delimiter='\t', skip_header=True)

class majority_vote_classifier:
    def __init__(self)-> None:
        self.majority = None
    
    def fit(self, data):
        label = data[:,-1]
        self.majority = int(np.sum(label==1)>=np.sum(label==0))
    
    def predict(self, entry):
        return self.majority


def write_to_output(data, classifier, output_file_path):
    with open(output_file_path, 'w') as file:
        for entry in data:
            p_label = classifier.predict(entry)
            file.write(f"{p_label}\n")



def write_error(input_train_data, input_test_data, output_train_path, output_test_path, error_file_path):
    true_train_label = input_train_data[:, -1]
    true_test_label = input_test_data[:, -1]
    with open(output_train_path, 'r') as file:
        lines = file.readlines()
        pred_train_label = [int(line.strip()) for line in lines]
        pred_train_label = np.array(pred_train_label)

    with open(output_test_path, 'r') as file:
        lines = file.readlines()
        pred_test_label = [int(line.strip()) for line in lines]
        pred_test_label = np.array(pred_test_label)
    
    train_error = np.sum(true_train_label != pred_train_label)/len(true_train_label)
    test_error = np.sum(true_test_label != pred_test_label)/len(true_test_label)

    with open(error_file_path, 'w') as file:
        file.write(f"error(train): {train_error }\n")
        file.write(f"error(test): {test_error }\n")



if __name__ == '__main__':
    if len(sys.argv)==6:
        train_infile = sys.argv[1]
        test_infile = sys.argv[2]   
        train_outfile = sys.argv[3]
        test_outfile = sys.argv[4]
        metric_outfile = sys.argv[5]

    # convert tsv to np data 
    np_train_data = tsv_to_numpy(train_infile)
    np_test_data = tsv_to_numpy(test_infile)

    # create and train the classifier on training data 
    classifier = majority_vote_classifier()
    classifier.fit(np_train_data)

    # write predictions 
    write_to_output(np_train_data, classifier, train_outfile)
    write_to_output(np_test_data, classifier, test_outfile)

    # write to metrics file
    write_error(np_train_data, np_test_data, train_outfile, test_outfile, metric_outfile)

    print(f"The train input file is: {train_infile}")
    print(f"The test input file is: {test_infile}")
    print(f"The  train output file is: {train_outfile}")
    print(f"The  test output file is: {test_outfile}")
    print(f"The  metric output file is: {metric_outfile}")