import os
import json
import argparse 

import numpy as np
from model_dsk import build_model,save_weights

data_dir='./data'
log_dir='./logs'

batch_size=16
seq_length=64

class TrainLogger(object):
    #to create the logs of the epochs.Object here is the log file name reference.
    def __init__(self,file):
        self.file=os.path.join(log_dir,file)
        self.epochs=0
        with open(self.file, 'w') as f:
            f.write('epoch,loss,acc\n')
            
    def add_entry(self, loss, acc):
        self.epochs += 1 #writing the log after every epoch
        s = '{},{},{}\n'.format(self.epochs, loss, acc)
        with open(self.file, 'a') as f:
            f.write(s)
            
            
def read_batches(T,vocab_size):
    length = T.shape[0] #129,665
    batch_chars = int(length / batch_size) # 8,104
    
    for start in range(0, batch_chars - seq_length, seq_length): # (0, 8040, 64)
            #for 1 batch 16 row and 64 consecutive time step columns are as follows:
            #We have (batch_chars/(batch_size x seq_length)) no of batches i.e. 126 batches in an epoch
            #for let 1st batch the the first row has consecutive characters from 0 to 63.for 2nd row in the 1st batch
            #the sequence starts from 8104 to 8104+63 .For 2nd batch the sequence for 1st row starts from 64 to 64+63 and
            #the sequence of character for 2nd row in 2nd batch starts from 8104+63+1 to 8104+63+1+63
        X = np.zeros((batch_size, seq_length)) # 16X64
        #X is an input tensor
        Y = np.zeros((batch_size, seq_length, vocab_size))# 16X64X86
        #Y is an output tensor (3rd dimension as one hot encoding )
        for batch_idx in range(0, batch_size):
            for i in range(0, seq_length):
                X[batch_idx, i] = T[batch_chars * batch_idx + start + i] # 
                Y[batch_idx, i, T[batch_chars * batch_idx + start + i + 1]] = 1
        yield X, Y




def train(text,epochs=100,save_freq=10):
    #train function with the input text ,epochs and model save frequency as arguments
    char_to_idx = { ch: i for (i, ch) in enumerate(sorted(list(set(text)))) }
    #assigning each individual character an id and storng it in a dictionary
    print("Number of unique characters: " + str(len(char_to_idx))) #86
    
    with open(os.path.join(data_dir, 'char_to_idx.json'), 'w') as f:
        json.dump(char_to_idx, f)
        
    idx_to_char = { i: ch for (ch, i) in char_to_idx.items() } #index to char 
    vocab_size = len(char_to_idx)
    
    
    model=build_model(batch_size,seq_length,vocab_size)
    #loading model
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    
    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32) #convert complete text into numerical indices
    print("Length of text:" + str(T.size)) #129,665 character_length
    
    steps_per_epoch = (len(text) / batch_size - 1) / seq_length
    
    log = TrainLogger('training_log.csv')
    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        
        losses, accs = [], []
        for i, (X, Y) in enumerate(read_batches(T, vocab_size)):
            
            print(X);

            loss, acc = model.train_on_batch(X, Y)
            print('Batch {}: loss = {}, acc = {}'.format(i + 1, loss, acc))
            losses.append(loss)
            accs.append(acc)
         
        log.add_entry(np.average(losses), np.average(accs))

        if (epoch + 1) % save_freq == 0:
            save_weights(epoch + 1, model)
            print('Saved checkpoint to', 'weights.{}.h5'.format(epoch + 1))
                           
                           
                           
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the model on some text.')
    parser.add_argument('--input', default='input.txt', help='name of the text file to train from')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--freq', type=int, default=10, help='checkpoint save frequency')
    args = parser.parse_args()

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    train(open(os.path.join(data_dir, args.input)).read(), args.epochs, args.freq)

