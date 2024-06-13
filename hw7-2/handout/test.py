import torch

# # Simulate a random tensor with shape (batch_size, seq_length, target_size)
# # Let's say we have a batch of 2 sequences, each of length 3, and 4 possible target tags (target_size)
# batch_size, seq_length, target_size = 2, 3, 4
# tag_dist = torch.randn(batch_size, seq_length, target_size)
# print(tag_dist)
# # Apply torch.max along the target_size dimension to find the predicted tag indices
# max_values, predicted_tags = torch.max(tag_dist, dim=2)
# print(max_values)

# print("tags:{}".format(predicted_tags))

from rnn import TextDataset


# train_input =  "./data/en.train_10.twocol.oov" 
# test_input = "./data/en.val_10.twocol.oov"
# word_to_idx = {}
# tag_to_idx = {}
# idx_to_tag = {}
# train_dataloader = TextDataset(train_input, word_to_idx, tag_to_idx, idx_to_tag)
# test_dataloader = TextDataset(test_input, word_to_idx, tag_to_idx, idx_to_tag)
# train_sent_out = 'train_sentences_out.txt'
# train_tags_out = 'train_tags_out.txt'


# with open(train_sent_out, 'w') as t_s:
#   with open(train_tags_out, 'w') as t_t:
#     for i, (sentences, tags) in enumerate(train_dataloader):
#       t_s.write(str(sentences))
#       t_s.write('\n')
#       t_t.write(str(tags))
#       t_t.write('\n')


# test_sent_out = 'test_sentences_out.txt'
# test_tags_out = 'test_tags_out.txt'


# with open(test_sent_out, 'w') as t_s:
#   with open(test_tags_out, 'w') as t_t:
#     for i, (sentences, tags) in enumerate(test_dataloader):
#       t_s.write(str(sentences))
#       t_s.write('\n')
#       t_t.write(str(tags))
#       t_t.write('\n')

import torch
import numpy as np


tensor = torch.tensor([4, 5, 6])

# Convert tensor to numpy array for demonstration
new_elements = tensor.numpy()

# Original list for demonstration
existing_list = [1, 2, 3]

# Method 1: Using `+` Operator
result_plus_operator = existing_list + new_elements.tolist()

# Method 2: Using `.extend()` Method
result_extend_method = existing_list.copy()  # Copy to not modify the original
result_extend_method.extend(new_elements)

# Method 3: Using `list()` and `+` Operator
result_list_plus = existing_list + list(new_elements)

# Method 4: Using `np.concatenate()` for Numpy Arrays
existing_array = np.array(existing_list)
result_concatenate = np.concatenate((existing_array, new_elements))



print(result_plus_operator)
print(result_extend_method)
print(result_list_plus)
print(result_concatenate)
# print(list_using_plus)
# print(list_using_extend)
# print(list_direct_append)


tensor_e = torch.randn(2, 3, 4)
flattened_tensor = tensor_e.flatten()
print(flattened_tensor.tolist())

clist = existing_list.copy() 
clist.extend(flattened_tensor.tolist())
print(clist)
print(len(clist)) 