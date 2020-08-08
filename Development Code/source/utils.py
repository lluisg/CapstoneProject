
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
    LIMIT_WORDS = 500

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
            if len(word) <= len('supercalifragilisticexpialidocious') and idx <= LIMIT_WORDS:
                words.append(word)

    return words


def word2integer(letter_dict, word, pad = 34):
    padding = 0 #we will pad with zeros
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
    if len(word_padded) < pad:
        word_padded = (word_padded + pad * [padding])[:pad]

    return word_padded, length

def sentence2integer(letter_dict, data, pad = 34):
    result = []
    lengths = []
    perc = 0

    for idx_w, word in enumerate(data):

        if idx_w / len(data) >= perc:
            print('{} / {} word = {}%'.format(idx_w, len(data), np.round(perc*100, decimals = 1)))
            perc = perc+0.1

        converted_word, len_word = word2integer(letter_dict, word, pad)
        result.append(converted_word)
        lengths.append(len_word)

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


def integer2word(number_dict, word, len_word):

    word_back = []
    for letter_index, letter in enumerate(word):
        if letter_index < len_word:
            if letter in number_dict:
                word_back.append(number_dict[letter])
            else:
                word_back.append('.')

    return word_back


def integer2sentence(number_dict, data, len_data):
    result = []

    perc=0
    idx_w = 0
    for word, len_w in zip(data, len_data):
        if idx_w / len(data) >= perc:
            print('{} / {} words = {}%'.format(idx_w, len(data), np.round(perc*100, decimals = 1)))
            perc = perc+0.1

        converted_word = integer2word(number_dict, word, len_w)

        result.append(converted_word)
        idx_w += 1

    return result


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
    integer_sentence, len_sentence = sentence2integer(dictionary, input_data_words)
    print('Jumbling data')
    jumbled_sentence = jumble_data(integer_sentence, len_sentence)

    #words with the length previously
    for word, leng in zip(jumbled_sentence, len_sentence):
        word.insert(0, leng)

    return integer_sentence, jumbled_sentence, len_sentence

def read_prediction(prediction, lengths, dictionary):
    return join_sentence(convert_back_data(dictionary, prediction, lengths))
