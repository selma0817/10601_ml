import csv
import numpy as np
import argparse

VECTOR_LEN = 300   # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt



## we rely on word embedding for feature transform 


################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map

def trim(dataset, glove_map):
    ret_dataset = []
    for review in dataset:
        label = review[0]
        comment = review[1].split(" ")
        temp = [w for w in comment if w in glove_map.keys()]
        ret_dataset.append((label, temp))
    return ret_dataset

def word_embed(t_dataset, glove_map):
    v_dataset = []
    for item in t_dataset:
        size = len(item[1])
        glove_v = np.zeros(VECTOR_LEN)
        for word in item[1]:
            glove_v+=glove_map[word]
        glove_v = glove_v/size
        v_dataset.append([item[0], glove_v.tolist()])
    return v_dataset

def write_output(v_dataset, out_path):
     with open(out_path, 'w') as file:
        for item in v_dataset:
            file.write("{:.6f}\t".format(item[0]))
            for val in item[1]:
                file.write("{:.6f}\t".format(val))
            file.write("\n")
        file.close()


def pipeline_write_output(train_data, val_data, test_data, train_path, val_path, test_path):
    write_output(train_data, train_path)
    write_output(val_data, val_path)
    write_output(test_data, test_path)





if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str, help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str, help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str, help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str, help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    # prepare the dataset 
    dataset_train = load_tsv_dataset(args.train_input)
    dataset_val = load_tsv_dataset(args.validation_input)
    dataset_test = load_tsv_dataset(args.test_input)

    # create glove_map
    glove_map = load_feature_dictionary(args.feature_dictionary_in)

    # trim datasets 
    t_train = trim(dataset_train, glove_map)
    t_val = trim(dataset_val, glove_map)
    t_test = trim(dataset_test, glove_map)

    # calc embed 
    embed_train = word_embed(t_train, glove_map)
    embed_val = word_embed(t_val, glove_map)
    embed_test = word_embed(t_test, glove_map)

    # write output
    pipeline_write_output(embed_train, embed_val, embed_test, args.train_out, 
    args.validation_out, args.test_out)

    # output format is label\tvalue1\tvalue2\tvalue3 ...\tvalue300\n