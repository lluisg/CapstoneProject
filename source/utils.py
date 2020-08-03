
import re
from bs4 import BeautifulSoup

import pickle

import os
import glob

from itertools import groupby
import unicodedata
import string

import numpy as np

def splitWithIndices(s, c=' '):
    p = 0
    for k, g in groupby(s, lambda x:x==c):
        q = p + sum(1 for i in g)
        if not k:
            yield p, q # or p, q-1 if you are really sure you want that
        p = q

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')


def post_to_words(post):
    post = post.lstrip().rstrip()
    post = BeautifulSoup(post, "html.parser").get_text()
    post = post.lower()
    
    words_accents = []
    for word in post:
        words_accents.append(strip_accents(word))
        
    words_ind = splitWithIndices(words_accents)
        
    words_aux = []
    for idx in words_ind:
        words_aux.append(post[idx[0]:idx[1]])
    
    words = []
    for idx, word in enumerate(words_aux):
        isvalid = True
        isdigit = True
        ispunct = True

        for letter in word:
            if not letter.isdigit():
                if letter not in string.punctuation:
                    if not letter.isalpha():
                        isvalid = False
            
            if not letter.isdigit():
                isdigit = False
            if letter not in string.punctuation:
                ispunct = False
                
        if isvalid == True and isdigit == False and ispunct == False:
            if len(word) <= len('supercalifragilisticexpialidocious') and idx <= 500:
                words.append(word)
        
    return words

def convert_and_pad(letter_dict, word, pad = 34):
    NOLETTER = 0 # We will use 0 to represent the 'no letter' category
    
    word_padded = []
    length = len(word)
    
    #conversion
    for letter_index, letter in enumerate(word):
        if letter in letter_dict:
            if letter_dict[letter] >= 0:
                word_padded.append(letter_dict[letter])
        else:
            length -= 1
    
    #padding
    if len(word_padded) < 34:
        word_padded = (word_padded + pad * [NOLETTER])[:pad]
            
    return word_padded, length

def convert_and_pad_data(letter_dict, data, pad = 34):
    
    result = []
    lengths = []

    perc = 0        
    
    for idx_w, word in enumerate(data):
        
        if idx_w / len(data) >= perc:
            print('{} / {} word = {}%'.format(idx_w, len(data), np.round(perc*100, decimals = 1)))
            perc = perc+0.1
        
#         print('word------>', word)
        converted_word, len_word = convert_and_pad(letter_dict, word, pad)
#         print('word------>', word, leng_word)
#         print('word converted--->', converted_word, leng_word)

        result.append(converted_word)
        lengths.append(len_word)
        
#     return np.array(result), np.array(lengths)
    return result, lengths

def jumble_word(word, word_len):
    if word_len <= 2:
        word_j = word
        
    else:
        sub_word = []
        for w in word:
            sub_word.append(w)

        sub_word = sub_word[1:word_len-1]
        shufled_word = shuffle(sub_word)
        
        word_j = word.copy()
        word_j[1:word_len-1] = shufled_word
        
    return word_j

def jumble_data(data, data_len):
    jumbled_data = []
    
    idx_w = 0
    perc = 0
    for word, w_len in zip(data, data_len):
        jumbled_sentence = []

        if idx_w / len(data) >= perc:
            print('{} / {} words = {}%'.format(idx_w, len(data), np.round(perc*100, decimals = 1)))
            perc = perc+0.1

        jumbled_word = jumble_word(word, w_len)

        jumbled_data.append(jumbled_word)
        idx_w+=1
        
    return jumbled_data

def one_hot_encode(sequence, dict_size, seq_len, batch_size):
    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((batch_size, seq_len, dict_size), dtype=np.float64)
    
    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(batch_size):
        for u in range(seq_len):
            features[i, u, sequence[i][u]] = 1
    return features

def convert_back_and_unpad(number_dict, word):
    
    word_back = []
    for letter_index, letter in enumerate(word):
        if letter in number_dict:
            word_back.append(number_dict[letter])
            
    if len(word_back) == 0:
        word_back.append('.')
            
    return word_back


def convert_back_data(number_dict, data):
    result = []
    
    perc=0
    for idx_w, word in enumerate(data):
        if idx_w / len(data) >= perc:
            print('{} / {} words = {}%'.format(idx_w, len(data), np.round(perc*100, decimals = 1)))
            perc = perc+0.1


#         print('word------>', word)
        converted_word = convert_back_and_unpad(number_dict, word)
#         print('word converted--->', converted_word, leng_word)

        result.append(converted_word)
        
#     return np.array(result), np.array(lengths)
    return result

# We will use the input training data to check and we will see how the sentence would be seen once jumbled
def join_sentence(sentence):
    list_words = []
    for word in sentence:
        w = ''.join(word)
        list_words.append(w)

    sentence_joined = ' '.join(list_words)
    
    return sentence_joined

def prepare_predict(sentence, dictionary):
    input_data_words = post_to_words(sentence)
    
    print('Converting data')
    integer_sentence, len_sentence = convert_and_pad_data(dictionary, input_data_words)
    print('Jumbling data')
    jumbled_sentence = jumble_data(integer_sentence, len_sentence)
        
    return integer_sentence, jumbled_sentence
