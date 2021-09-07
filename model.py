import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dropout, TimeDistributed, Dense, Activation, Embedding
#importing necessary keras modules for char-RNN modelling

def build_model(batch_size,seq_length,vocab_size):
    #batch_size ,sequence of length and the total vocab of characters of the corpus as an argument
    model=Sequential() # Sequential model initialization
    model.add(Embedding(vocab_size,512,batch_input_shape=(batch_size,seq_length)))
    #Embedding the input character (one from 86 unique character in our corpus into 512 dimension vector.
    #batch_input_shape is  batch_size(16 in our case) x seq_length(64 in our case) which mkaes the total 
    #input tensor of shape 16 x 64 x 512
    for i in range(3):
        model.add(LSTM(256, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))
    #Adding 3 lstm layers consecutively with dropout layer after each LSTM layer.
    #The LSTM cell size taken is 256. return_sequences=True gives the predicted yi for every input xi in that very timestep
    #unlike in other many-to-many seq2seq networks where the input sequence is first fed altogether for various timestep 
    #and then the yi is predicted as a sequence for the whole input.
    #Stateful=true passes the state value of the previous batch same row last outut value (as they are continuous) to
    #the first layer of the next batch of the same row as a state value of the previous state.This helps us achieving 
    #larger sequences.
    #Dropout with 20% of the weights as the data points are less to avoid overfitting.
    model.add(TimeDistributed(Dense(vocab_size)))
    #Time Distributed dense value is for the generation of predicted yi at each time step .
    #The Dense layer if added at last like in every other cases then the model will not be able to understand the sequence.
    model.add(Activation('softmax'))
    return model
    #the softmax layer because the predicted yi is a plausible vector (one hot encoded vector) for the size of the vocab_size
    #thus yi has dimension as batch_size x sequence length x vocab_size(one-hot encoded)
    
model_dir='./model'    #path to save the model weights   
def save_weights(epoch,model):
    #function to save weights of the trained model
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        model.save_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch)))
        #saving models epoch wise
def load_weights(epoch,model):
    #function to load weights
    model.load_weights(os.path.join(model_dir, 'weights.{}.h5'.format(epoch)))
    
if __name__ == '__main__':
    model = build_model(16, 64, 50) #batch_size=16,sequence_length=64,vocab_size
    model.summary()

