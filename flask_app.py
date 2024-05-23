from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Initialize Flask app
app = Flask(__name__)

# Define the actions and label map
actions = np.array(['JumpingJacks', 'No JumpingJacks'])
label_map = {label: num for num, label in enumerate(actions)}

# Function to create the model
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

# Create and load the model
model = create_model()
model.load_weights(r"C:\Graduation project\jumping\models\weights1\jumpingjacks.h5")

# Define the process_sequence function
def process_sequence(sequence):
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    if res[np.argmax(res)] > 0.5:
        action = actions[np.argmax(res)]
        return action
    else:
        return None

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sequence = np.array(data['sequence'])
    action = process_sequence(sequence)
    response = {
        'action': action
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
