from flask import Flask, request, jsonify
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize sequence and constants
SEQUENCE_LENGTH = 30
sequence = []

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
model.load_weights("jumpingjacks.h5")

def extract_landmarks(results):
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
    return landmarks

def process_sequence(sequence):
    res = model.predict(np.expand_dims(sequence, axis=0))[0]
    if res[np.argmax(res)] > 0.5:
        action = actions[np.argmax(res)]
        return action
    else:
        return None

@app.route('/detect', methods=['POST'])
def detect():
    global sequence
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    file_bytes = np.fromfile(file, np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    # Convert frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Extract landmarks
    keypoints = extract_landmarks(results)
    sequence.append(keypoints)
    if len(sequence) > SEQUENCE_LENGTH:
        sequence.pop(0)

    action = None
    if len(sequence) == SEQUENCE_LENGTH:
        action = process_sequence(sequence)

    response = {
        'action': action
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug = False,use_reloader=False,host = "0.0.0.0")
