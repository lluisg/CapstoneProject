import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

# from utils import *
from utils import join_sentence, convert_back_data, one_hot_encode, prepare_predict

def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['embedding_dim'], model_info['hidden_dim'], model_info['vocab_size'])

    # Load the store model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the saved word_dict.
    word_dict_path = os.path.join(model_dir, 'word_dict.pkl')
    with open(word_dict_path, 'rb') as f:
        model.word_dict = pickle.load(f)

    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == 'text/plain':
        data = serialized_input_data.decode('utf-8')
        return data
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    return str(prediction_output)

def predict_fn(input_data, dictionary, model):
    print('Inferring sentiment of input data.')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model.word_dict is None:
        raise Exception('Model has not been loaded properly, no word_dict.')
    
    # TODO: Process input_data so that it is ready to be sent to our model.
    #       You should produce two variables:
    #         data_X   - A sequence of length 34 which represents the converted word
    #         data_len - The real length of the word
    
    if isinstance(data_input[0], str):
        print('Is an string sentence input')
        original_s_int, jumbled_s_int = prepare_predict(s, dictionary)
    else:
        print('Is an integer sentence input')
        jumbled_s_int = data_input.copy()    
    
    integer_sentence = [] 
    for word in jumbled_s_int:
#         print('word: ', word)
        word_batch = [word]

        if len(word) > 3:
            dict_size = 27 #including the 0
            seq_len = 34
            batch_size =1
            test_seq = one_hot_encode(word_batch, dict_size, seq_len, batch_size)
            

            data = torch.from_numpy(test_seq).float().squeeze().to(device)
            # Have the torch as a batch of size 1
            data_batch = data.view(1, np.shape(data)[0], np.shape(data)[1])
#             print('size: {} -> {}'.format(np.shape(data), np.shape(data_batch)))
            # Make sure to put the model into evaluation mode
            model.eval()

            with torch.no_grad():
                
                output = model.forward(data_batch)
                
                word_integer = []
                for letter in output[0]: #as there's only 1 batch
                    
#                     print('letter', np.shape(letter))
                    letter_numpy = letter.numpy()
#                     print('numpy', type(letter_numpy), np.shape(letter_numpy))
                    
                    max_value_ind = np.argmax(letter_numpy, axis=0)
#                     print(max_value_ind)
#                     print(letter_numpy)



                    word_integer.append(max_value_ind)
                
        else:

            word_integer = word_batch.copy()
            
        integer_sentence.append(word_integer)
#         print('integer word: ', word_integer) 
    print('Convert back sentences')
    string_sentence = join_sentence(convert_back_data(int2letter, integer_sentence))
    
    return integer_sentence, string_sentence