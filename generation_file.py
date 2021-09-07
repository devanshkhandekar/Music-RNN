import argparse
import os
import json

import numpy as np

from model_dsk import build_model, load_weights

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding

data_dir = './data'
model_dir = './model'

def build_sample_model(vocab_size):
    model = Sequential()
    model.add(Embedding(vocab_size, 512, batch_input_shape=(1, 1)))
    #input shape is a single character that we provide as an argument
    for i in range(3):
        model.add(LSTM(256, return_sequences=(i != 2), stateful=True))
        #return sequence =False for the final layer since we don't need the 
        #return_sequences returns the hidden state output for each input time step. Each LSTM cell will output one hidden state h for each input time step. If we have a single input sequence with 10-time steps, then we have 10 hidden state values for each time step in the single input. But we can access these values, if and only if provide LSTM parameter "return_sequence = True". You must set return_sequences=True when stacking LSTM layers so that the second LSTM layer has a three-dimensional sequence input. Return sequence =True  is used when we are connecting two LSTM layers. So in that case every timestep's output is connected to next layer's input accordingly.  But here we don't need that in final layer.the last layer will generate the output as a many to one rnn does but the transmission will be false as there is no layer to transmit to.
        model.add(Dropout(0.2))

    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    return model

def sample(epoch, header, num_chars):
    #Header is the random character to initialise with and num_chars is the num of characters we want
    #our model to generate the sequence of characters.The default length set is 512 for num_chars.
    with open(os.path.join(data_dir, 'char_to_idx.json')) as f:
        char_to_idx = json.load(f)
    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() }
    vocab_size = len(char_to_idx)

    model = build_sample_model(vocab_size)
    load_weights(epoch, model)
    model.save(os.path.join(model_dir, 'model.{}.h5'.format(epoch)))

    sampled = [char_to_idx[c] for c in header]
    print(sampled)
    

    for i in range(num_chars):
        batch = np.zeros((1, 1))
        if sampled:
            batch[0, 0] = sampled[-1]
        else:
            batch[0, 0] = np.random.randint(vocab_size)
        result = model.predict_on_batch(batch).ravel()
        sample = np.random.choice(range(vocab_size), p=result)
        sampled.append(sample)

    return ''.join(idx_to_char[c] for c in sampled)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample some text from the trained model.')
    parser.add_argument('epoch', type=int, help='epoch checkpoint to sample from')
    parser.add_argument('--seed', default='', help='initial seed for the generated text')
    parser.add_argument('--len', type=int, default=512, help='number of characters to sample (default 512)')
    args = parser.parse_args()

    print(sample(args.epoch, args.seed, args.len))
