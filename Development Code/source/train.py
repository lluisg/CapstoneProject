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

from model import WordOrderer, Decoder, Encoder

def obtain_data(data_dir, namefile, batch_s):
    # Load the training data.
    train_sample = pd.read_csv(os.path.join(data_dir, namefile), header=None, names=None)
    print('Loaded csv')

    train_sample_y = train_sample[train_sample.columns[0:34]]
    train_sample_len = train_sample[train_sample.columns[34]]
    train_sample_X = train_sample[train_sample.columns[34:69]]
    print('Size read from csv -> X: {}, Y: {}, len: {}'.format(train_sample_X.shape, train_sample_len.shape, train_sample_y.shape))

    X_np = train_sample_X.to_numpy(copy=True)
    len_np = train_sample_len.to_numpy(copy=True)
    Y_np = train_sample_y.to_numpy(copy=True)
    print('To numpied')

    dict_size = 34
    seq_len = 35
    batch_size = len(train_sample_X)
    input_seq = one_hot_encode(X_np, dict_size, seq_len, batch_size)
    print('One hot encoded')

    train_torch_x = torch.from_numpy(input_seq).float().squeeze()
    train_torch_len = torch.from_numpy(len_np).float().squeeze().type(torch.long)
    train_torch_y = torch.from_numpy(Y_np).float().squeeze().type(torch.long)
    print('Torched')

    train_sample_ds = torch.utils.data.TensorDataset(train_torch_x, train_torch_y, train_torch_len)
    train_loader = torch.utils.data.DataLoader(train_sample_ds, batch_size=batch_s)
    print('Train loaded')

    return train_loader


def train(model, train_loader, epochs, optimizer, loss_fn, device):
    print('Start training')
    total_length = len(train_loader.dataset)
    model.train()
    loss_return = []
    increased = 0
    loss_previous = 0

    for epoch in range(1, epochs + 1):
        batchs_done = 0
        total_loss = 0

        for batch in train_loader:
            batch_X, batch_y, batch_len = batch
            len_batch = len(batch_X)

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            out = model(batch_X)

            batch_loss = 0
            for result, target, len_word in zip(out, batch_y, batch_len):
                loss = loss_fn(result[:len_word, :], target[:len_word])
                batch_loss += loss

            batch_loss.backward()
            optimizer.step()

            total_loss += batch_loss.data.item()
            batchs_done += len_batch
#             print('Batch done. {} / {} inputs = {}%'.format(
#                 batchs_done, total_length, np.round(batchs_done/total_length*100, decimals = 1)))

        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))
        loss_return.append(total_loss / len(train_loader))

        # early stopping
        if total_loss > loss_previous:
            increased += 1
            print('Increased ({})'.format(increased))
        else:
            increased = 0

        loss_previous = total_loss

        if increased >= 3:
            print('Early stopping!')
            break
    return loss_return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # SageMaker Parameters
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    # Training Parameters
    parser.add_argument('--batch_size', type=int, default=128, metavar='B',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=10, metavar='E',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
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
    with open(os.path.join(args.data_dir, 'int2letter_dict.pkl'), "rb") as f:
        int2letter = pickle.load(f)

    train_loader = obtain_data(args.data_dir, 'train.csv', args.batch_size)

    # Build the model.
    encoder = Encoder(args.input_dim, args.hidden_dim)
    decoder = Decoder(args.input_dim, args.output_dim, args.hidden_dim)
    model = WordOrderer(encoder, decoder).to(device)
    print('Models')

    with open(os.path.join(args.data_dir, 'int2letter_dict.pkl'), "rb") as f:
        model.letter2int_dict = pickle.load(f)
    with open(os.path.join(args.data_dir, 'letter2int_dict.pkl'), "rb") as f:
        model.int2letter_dict = pickle.load(f)


    print("Model loaded with input_dim {}, output_dim {}, hidden_dim {}.".format(args.input_dim, args.output_dim, args.hidden_dim))

    # Train the model.
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Going to train')
    train(model, train_loader, args.epochs, optimizer, loss_function, device)
    print('Trained')

    # Save the parameters used to construct the model
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_dim': args.input_dim,
            'output_dim': args.output_dim,
            'hidden_dim': args.hidden_dim,
        }
        torch.save(model_info, f)

	# Save the two letter2int_dict
    letter2int_dict_path = os.path.join(args.model_dir, 'int2letter_dict.pkl')
    with open(letter2int_dict_path, 'wb') as f:
        pickle.dump(model.letter2int_dict, f)

    letter2int_dict_path2 = os.path.join(args.model_dir, 'letter2int_dict.pkl')
    with open(letter2int_dict_path2, 'wb') as f:
        pickle.dump(model.int2letter_dict, f)

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
