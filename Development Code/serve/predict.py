import argparse
import os
import sys
import pandas as pd
import numpy as np
import pickle
import sagemaker_containers

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F

from sklearn.utils import shuffle

# import model
from model import WordOrderer, Encoder, Decoder
from utils import join_sentence, integer2sentence, one_hot_encode, prepare_predict, read_prediction

def model_fn(model_dir):
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = Encoder(model_info['input_dim'], model_info['hidden_dim'])
    decoder = Decoder(model_info['input_dim'], model_info['output_dim'], model_info['hidden_dim'])
    model = WordOrderer(encoder, decoder).to(device)

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # Load the 2 saved letter2int_dict.
    letter2int_dict_path = os.path.join(model_dir, 'letter2int_dict.pkl')
    with open(letter2int_dict_path, 'rb') as f:
        model.letter2int_dict = pickle.load(f)
    int2letter_dict_path = os.path.join(model_dir, 'int2letter_dict.pkl')
    with open(int2letter_dict_path, 'rb') as f:
        model.int2letter_dict = pickle.load(f)

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

def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model.letter2int_dict is None:
        raise Exception('Model has not been loaded properly, no letter2int_dict.')
    if model.int2letter_dict is None:
        raise Exception('Model has not been loaded properly, no int2letter_dict.')

    original_s_int, jumbled_s_int, data_len = prepare_predict(input_data, model.letter2int_dict)

    integer_sentence = []
    for word in jumbled_s_int:
        word_batch = [word]

        if len(word) > 3:
            dict_size = 34
            seq_len = 35
            batch_size =1
            test_seq = one_hot_encode(word_batch, dict_size, seq_len, batch_size)

            data = torch.from_numpy(test_seq).float().squeeze().to(device)
            # Have the torch as a batch of size 1
            data_batch = data.view(1, np.shape(data)[0], np.shape(data)[1])

            # Make sure to put the model into evaluation mode
            model.eval()
            with torch.no_grad():
                output = model.forward(data_batch)
                word_integer = []

                for letter in output[0]: #as there's only 1 batch
                    letter_numpy = letter.numpy()
                    max_value_ind = np.argmax(letter_numpy, axis=0)
                    word_integer.append(max_value_ind)
        else:
            word_integer = word_batch.copy()

        integer_sentence.append(word_integer)

    output_string = read_prediction(integer_sentence, data_len, model.int2letter_dict)
    return output_string
