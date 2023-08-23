import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv3D, LSTM, Dense, Dropout, Bidirectional, 
    MaxPool3D, Activation, Reshape, SpatialDropout3D,
    BatchNormalization, TimeDistributed, Flatten
)

def load_model() -> Sequential:
    """
    Create and load a pre-trained Sequential model.
        Returns:
        - Sequential: A pre-trained Keras Sequential model.
    """
    
    # Initialize a Sequential model
    model = Sequential()

    # Add a 3D convolution layer with 128 filters, a kernel size of 3, and padding set to 'same'
    model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
    
    # Add ReLU activation function
    model.add(Activation('relu'))
    
    # Add a 3D max pooling layer with a pool size of (1, 2, 2)
    model.add(MaxPool3D((1, 2, 2)))

    # Repeat the pattern: Convolutional layer -> Activation -> Max pooling for the subsequent layers
    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1, 2, 2)))

    # Flatten the output for each time step before feeding it to the LSTM layer
    model.add(TimeDistributed(Flatten()))

    # Add two bidirectional LSTM layers, each followed by a dropout layer to prevent overfitting
    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(0.5))

    # Add a dense layer with 41 units (possible classes) with softmax activation for classification
    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    # Load the pre-trained weights from a checkpoint
    model.load_weights(os.path.join('..', 'models', 'checkpoint'))

    return model
