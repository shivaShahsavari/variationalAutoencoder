from keras import objectives, backend as K
from keras.layers import Bidirectional, Dense, Embedding, Input, Lambda, LSTM, RepeatVector, TimeDistributed
from keras.models import Model
import keras
import tensorflow as tf

class VAE(keras.Model):
    def __init__(self,**kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = None
        self.decoder = None
        self.predictor=None
        self.autoencoder=None
        self.z_mean=None
        self.z_log_var=None
    
    def create(self,vocab_size=500, max_length=300, latent_rep_size=200):

        x = Input(shape=(max_length,))
        x_embed = Embedding(vocab_size, 32, input_length=max_length)(x)

        self.z_mean,self.z_log_var,encoded = self._build_encoder(x_embed, latent_rep_size=latent_rep_size, max_length=max_length)
        self.encoder = Model(inputs=x, outputs=encoded, name="encoder")
        self.encoder.summary()

        encoded_input = Input(shape=(latent_rep_size,))
        predicted = self._build_predictor(encoded_input)
        self.predictor = Model(encoded_input, predicted)
        
        decoded = self._build_decoder(encoded_input, vocab_size, max_length)
        self.decoder = Model(encoded_input, decoded, name="decoder")
        self.decoder.summary()

        x_decoded=self._build_decoder(encoded, vocab_size, max_length)
        x_predicted=self._build_predictor(encoded)
        self.autoencoder = Model(inputs=x, outputs=[x_decoded,x_predicted])
        self.autoencoder.compile(optimizer='Adam', loss={'decoded_mean':tf.keras.losses.KLDivergence(),'pred':'binary_crossentropy'}, metrics=['accuracy'])


    def _build_encoder(self, x_embed, latent_rep_size=200, max_length=300, epsilon_std=0.01):
        h = Bidirectional(LSTM(500, return_sequences=True, name='lstm_1'), merge_mode='concat')(x_embed)
        h = Bidirectional(LSTM(500, return_sequences=False, name='lstm_2'), merge_mode='concat')(h)
        h = Dense(435, activation='relu', name='dense_1')(h)

        def sampling(args):
            z_mean_, z_log_var_ = args
            batch_size = K.shape(z_mean_)[0]
            epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., stddev=epsilon_std)
            return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

        z_mean = Dense(latent_rep_size, name='z_mean', activation='linear')(h)
        z_log_var = Dense(latent_rep_size, name='z_log_var', activation='linear')(h)

        return (z_mean,z_log_var,Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))


    def _build_decoder(self, encoded, vocab_size, max_length):
        repeated_context = RepeatVector(max_length)(encoded)
        h = LSTM(500, return_sequences=True, name='dec_lstm_1')(repeated_context)
        h = LSTM(500, return_sequences=True, name='dec_lstm_2')(h)
        decoded = TimeDistributed(Dense(vocab_size, activation='softmax'), name='decoded_mean')(h)
        return decoded

    def _build_predictor(self, encoded_input):
        h = Dense(100, activation='linear')(encoded_input)
        return Dense(1, activation='sigmoid', name='pred')(h)

###############################################################

from keras.callbacks import ModelCheckpoint
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
#from model import VAE
import numpy as np
import os

def main():

    vocab_size=500; max_length=300; latent_rep_size=200; epsilon_std=0.01
    NUM_WORDS = 1000
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=NUM_WORDS)

    print("Training data")
    print(X_train.shape)
    print(y_train.shape)

    print("Number of words:")
    print(len(np.unique(np.hstack(X_train))))

    X_train = pad_sequences(X_train, maxlen=max_length)
    X_test = pad_sequences(X_test, maxlen=max_length)

    train_indices = np.random.choice(np.arange(X_train.shape[0]), 2000, replace=False)
    test_indices = np.random.choice(np.arange(X_test.shape[0]), 1000, replace=False)

    X_train = X_train[train_indices] #<class 'numpy.ndarray'>
    y_train = y_train[train_indices] #<class 'numpy.ndarray'>

    X_test = X_test[test_indices] # <class 'numpy.ndarray'>
    y_test = y_test[test_indices] # <class 'numpy.ndarray'>

    print('X_train',X_train.shape)#(2000, 300)
    print('X_test',X_test.shape)#(1000, 300)

    temp = np.zeros((X_train.shape[0], max_length, NUM_WORDS))
    temp[np.expand_dims(np.arange(X_train.shape[0]), axis=0).reshape(X_train.shape[0], 1), np.repeat(np.array([np.arange(max_length)]), X_train.shape[0], axis=0), X_train] = 1
    X_train_one_hot = temp
    print('X_train_one_hot',X_train_one_hot.shape) #<class 'numpy.ndarray'>

    temp1 = np.zeros((X_test.shape[0], max_length, NUM_WORDS))
    temp1[np.expand_dims(np.arange(X_test.shape[0]), axis=0).reshape(X_test.shape[0], 1), np.repeat(np.array([np.arange(max_length)]), X_test.shape[0], axis=0), X_test] = 1
    x_test_one_hot = temp1  #<class 'numpy.ndarray'>
    print('x_test_one_hot',x_test_one_hot.shape)

    def create_model_checkpoint(dir, model_name):
        filepath = dir + '/' + \
                model_name + "-{epoch:02d}-{val_decoded_mean_acc:.2f}-{val_pred_loss:.2f}.h5"
        directory = os.path.dirname(filepath)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory)
        checkpointer = ModelCheckpoint(filepath=filepath,verbose=1,save_best_only=False)
        return checkpointer

                
    def train():
        vae = VAE()
        vae.create(vocab_size=NUM_WORDS, max_length=max_length)
        checkpointer = create_model_checkpoint('models', 'rnn_ae')
        vae.autoencoder.fit(x=X_train, y={'decoded_mean': X_train_one_hot, 'pred': y_train},batch_size=10, epochs=10,callbacks=[checkpointer],
                           validation_data=(X_test, {'decoded_mean': x_test_one_hot, 'pred':  y_test}))
     

    train()

    

if __name__=="__main__":
    main()

