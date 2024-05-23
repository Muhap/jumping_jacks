import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

actions = np.array(['JumpingJacks', 'No JumpingJacks'])

def create_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 132)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['categorical_accuracy'])
    return model

model = create_model()
model.load_weights(r"C:\Graduation project\jumping\models\weights1\jumpingjacks.h5")

sequence_buffer = []

def process_sequence(sequence):
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    if res[np.argmax(res)] > 0.5:
        return actions[np.argmax(res)]
    else:
        return None
