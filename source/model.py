import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim):
        super(Encoder, self).__init__()

        self.hid_dim = hid_dim
        self.rnn = nn.LSTM(input_dim, hid_dim, batch_first = True)
                
    def forward(self, src):
                
        #Encoder part                
#         print('shape input: ',  np.shape(src))
        outputs, (hidden, cell) = self.rnn(src)
#         print('(output) shape hidden: ',  np.shape(hidden))
#         print('(output) shape cell: ',  np.shape(cell))
        
        return hidden, cell
    
    
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, hid_dim):
        super(Decoder, self).__init__()

        self.hid_dim = hid_dim
        self.rnn = nn.LSTM(input_dim, hid_dim, batch_first = True)


        self.fc = nn.Linear(hid_dim, output_dim)

    def forward(self, src, hidden, cell):

#         print('shape input: ',  np.shape(src))
#         print('shape hidden: ',  np.shape(hidden))
#         print('shape cell: ',  np.shape(cell))
                        
        trg, (hidden, cell) = self.rnn(src, (hidden, cell))
#         print('(output) shape after LSTM: ',  np.shape(trg))

        # Reshaping the outputs such that it can be fit into the fully connected layer
        batch_size = np.shape(trg)[0]
#         print('Batch size: ',  batch_size)
        len_size = np.shape(trg)[1]
#         print('Length size: ',  len_size)

        out = trg.contiguous().view(-1, self.hid_dim)
#         print('FCL input shape: ',  np.shape(out))
        out = self.fc(out)
#         print('(output) shape after FCL: ',  np.shape(out))
        out = out.view(batch_size, len_size, -1)
#         print('(output) final shape: ',  np.shape(out))
        
        
        return out
    
class Seq2Seq(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.word_dict = None
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        
    def forward(self, src):

#         print('Encoder:')
        hidden, cell = self.encoder(src)
        
#         print('Decoder:')
        trg = self.decoder(src, hidden, cell)
        
#         print('Step part:')
#         letter_scores = F.log_softmax(trg)
#         print('shape letter scores: ', np.shape(letter_scores))
        
#         return letter_scores
        return trg
    