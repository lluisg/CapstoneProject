import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.lstm = nn.LSTM(input_dim, hid_dim, batch_first = True)

    def forward(self, src):
        outputs, (hidden, cell) = self.lstm(src)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim):
        super(Decoder, self).__init__()
        self.hid_dim = hid_dim
        self.lstm = nn.LSTM(output_dim, hid_dim, batch_first = True)
        self.output_dim = output_dim
        self.fc = nn.Linear(hid_dim, output_dim)

    def forward(self, src, hidden, cell):
        trg, (hidden, cell) = self.lstm(src, (hidden, cell))
        # Reshaping the outputs such that it can be fit into the fully connected layer
        batch_size = np.shape(trg)[0]
        len_size = np.shape(trg)[1]

        out = trg.contiguous().view(-1, self.hid_dim)
        out = self.fc(out)
        out = out.view(batch_size, len_size, -1)
        return out

class WordOrderer(nn.Module):

    def __init__(self, encoder, decoder):
        super(WordOrderer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        assert encoder.hid_dim == decoder.hid_dim, "Encoder and Decoder don't have the same dimensions"

        self.letter2int_dict = None
        self.int2letter_dict = None

    def forward(self, jumbled):
        hidden, cell = self.encoder(jumbled)

        adapted = jumbled[:, 1:, :self.decoder.output_dim] #as we will not pass the first element to the Decoder
        # which was the length of the word, and so the vocabulary size gets reduced to 27
        word = self.decoder(adapted, hidden, cell)
        return word
