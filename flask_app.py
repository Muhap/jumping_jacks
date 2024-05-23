from flask import Flask, request, jsonify

import numpy as np

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
        #sequence_buffer.clear()  # Reset the buffer

    return jsonify({'action': action if action else 'Buffering'})

if __name__ == '__main__':
    app.run(debug=False)
