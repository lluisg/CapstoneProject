import argparse
import json
import os
import pickle
import sys
import sagemaker_containers
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import one_hot_encode

from model import Seq2Seq, Decoder, Encoder

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

    # Load the stored model parameters.
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

def _get_train_data_loader(batch_size, training_dir):
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_X = torch.from_numpy(train_data.drop([0], axis=1).values).long()

    train_ds = torch.utils.data.TensorDataset(train_X, train_y)

    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    optimizer    - The optimizer to use during training.
    loss_fn      - The loss function used for training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    total_length = len(train_loader.dataset)
    model.train()
    loss_return = []
    
    for epoch in range(1, epochs + 1):
        batchs_done = 0
        total_loss = 0
        
        for batch in train_loader:
            batch_X, batch_y, batch_len = batch
            len_batch = len(batch_X)
#             print('input shape X: {}, y:{}'.format(np.shape(batch_X), np.shape(batch_y)))
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            out = model(batch_X)
#             print('output shape: ',  np.shape(out))
#             print('target shape: ',  np.shape(batch_y))
            
            batch_loss = 0
            for result, target, len_word in zip(out, batch_y, batch_len):

                loss = loss_fn(result[:len_word, :], target[:len_word])
#                 loss.backward(retain_graph=True)
#                 optimizer.step()
                batch_loss += loss
    
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.data.item()
#             total_loss += batch_loss
        
            batchs_done += len_batch
#             print('Batch done. {} / {} inputs = {}%'.format(
#                 batchs_done, total_length, np.round(batchs_done/total_length*100, decimals = 1)))

        print("Epoch: {}, NLLLoss: {}".format(epoch, total_loss / len(train_loader)))
        loss_return.append(total_loss / len(train_loader))
        
    return loss_return

if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments when the script
    # is executed. Here we set up an argument parser to easily access the parameters.

    parser = argparse.ArgumentParser()
    
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    # Training Parameters
    parser.add_argument('--batch-size', type=int, default=512, metavar='N',
                        help='input batch size for training (default: 512)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')

    # Model Parameters
    parser.add_argument('--input_dim', type=int, default=27, metavar='ID',
                        help='size of the input dimension (default: 27)')
    parser.add_argument('--output_dim', type=int, default=27, metavar='OD',
                        help='size of the output dimension (default: 27)')
    parser.add_argument('--hidden_dim', type=int, default=64, metavar='HD',
                        help='size of the hidden dimension (default: 34)')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    
    #load dictionaries
    with open(os.path.join(args.data_dir, 'letter2int_dict.pkl'), "rb") as f:
        letter2int = pickle.load(f)
    with open(os.path.join(args.data_dir, 'letter2int_dict.pkl'), "rb") as f:
        int2letter = pickle.load(f)


    # Load the training data.
    train_sample = pd.read_csv(os.path.join(args.data_dir, 'train.csv'), header=None, names=None)
    print('Loaded csv')

    train_sample_y = train_sample[train_sample.columns[0:34]]
    train_sample_len = train_sample[train_sample.columns[34]]
    train_sample_X = train_sample[train_sample.columns[35:69]]
    print('Size read from csv -> X: {}, Y: {}, len: {}'.format(train_sample_X.shape, train_sample_len.shape, train_sample_y.shape))

    X_np = train_sample_X.to_numpy(copy=True)
    len_np = train_sample_len.to_numpy(copy=True)
    Y_np = train_sample_y.to_numpy(copy=True)
    print('To numpied')

    dict_size = 34
    seq_len = 34
    batch_size = len(train_sample_X)
    input_seq = one_hot_encode(X_np, dict_size, seq_len, batch_size)
    print('One hot encoded')

    train_torch_x = torch.from_numpy(input_seq).float().squeeze()
    train_torch_len = torch.from_numpy(len_np).float().squeeze().type(torch.long)
    train_torch_y = torch.from_numpy(Y_np).float().squeeze().type(torch.long)
    print('Torched')

    train_sample_ds = torch.utils.data.TensorDataset(train_torch_x, train_torch_y, train_torch_len)
    train_loader = torch.utils.data.DataLoader(train_sample_ds, batch_size=128)
    print('Train loaded')

#     train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    
    # Build the model.
    encoder = Encoder(args.input_dim, args.hidden_dim)
    decoder = Decoder(args.input_dim, args.output_dim, args.hidden_dim)
    model = Seq2Seq(encoder, decoder).to(device)

    with open(os.path.join(args.data_dir, 'letter2int_dict.pkl'), "rb") as f:
        model.word_dict = pickle.load(f)
        

    print("Model loaded with input_dim {}, output_dim {}, hidden_dim {}.".format(
        args.input_dim, args.output_dim, args.hidden_dim
    ))

    # Train the model.
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train(model, train_loader, args.epochs, optimizer, loss_function, device)

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'output_dim': args.output_dim,
            'hidden_dim': args.hidden_dim,
        }
        torch.save(model_info, f)

	# Save the word_dict
    word_dict_path = os.path.join(args.model_dir, 'letter2int_dict.pkl')
#     word_dict_path = os.path.join('data', 'cache', 'letter2int_dict.pkl')
    with open(word_dict_path, 'wb') as f:
        pickle.dump(model.word_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
