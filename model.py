from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Input, Conv1D
from keras.layers import GRU, BatchNormalization, TimeDistributed
import numpy as np
import matplotlib.pyplot as plt
from td_utils import graph_spectrogram
from pydub import AudioSegment

def model(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    
    X_input = Input(shape = input_shape)
    
    X = Conv1D(196, 15, strides=4)(X_input)      
    X = BatchNormalization()(X)             
    X = Activation('relu')(X)              
    X = Dropout(0.8)(X)          

    X = GRU(128, return_sequences=True)(X)         
    X = Dropout(0.8)(X)                  
    X = BatchNormalization()(X)               
    
    X = GRU(128, return_sequences=True)(X)  
    X = Dropout(0.8)(X)                        
    X = BatchNormalization()(X)                 
    X = Dropout(0.8)(X)                               
    
    X = TimeDistributed(Dense(1, activation='sigmoid'))(X)

    model = Model(inputs = X_input, outputs = X)
    
    return model  

def detect_triggerword(filename):
    
    plt.subplot(2, 1, 1)
    x = graph_spectrogram(filename)
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    
    return predictions

def chime_on_activate(filename, predictions, threshold, chime_file):
    
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]

    consecutive_timesteps = 0
    for i in range(Ty):
        consecutive_timesteps += 1
        if predictions[0, i, 0] > threshold and consecutive_timesteps > 75:
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')
    
def preprocess_audio(filename):
    
    padding = AudioSegment.silent(duration=10000)
    segment = AudioSegment.from_wav(filename)[:10000]
    segment = padding.overlay(segment)
    segment = segment.set_frame_rate(44100)
    segment.export(filename, format='wav')
