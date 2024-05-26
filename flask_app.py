import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from flask import Flask, request, jsonify


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
model.load_weights('jumpingjacks.h5')

sequence_buffer = []

def process_sequence(sequence):
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    if res[np.argmax(res)] > 0.5:
        return actions[np.argmax(res)]
    else:
        return None


app = Flask(__name__)

SEQUENCE_LENGTH = 30

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    keypoints = data['keypoints']
    keypoints = np.array(keypoints)

    if keypoints.shape != (132,):
        return jsonify({'error': 'Invalid keypoints shape, expected (132,)'}), 400
        
    global sequence_buffer  # Access the global sequence buffer variable
    # Append keypoints to the sequence buffer
    sequence_buffer.append(keypoints)
    if len(sequence_buffer) > SEQUENCE_LENGTH:
        sequence_buffer=sequence_buffer[-30:]  # Maintain the buffer size

    action = None
    if len(sequence_buffer) == SEQUENCE_LENGTH:
        action = process_sequence(sequence_buffer)
        sequence_buffer.clear()  # Reset the buffer

    return jsonify({'action': action if action else 'Buffering'})
    
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port=8001,use_reloader=False)
