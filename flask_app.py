from flask import Flask, request, jsonify
from model import process_sequence
import numpy as np

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sequence = data['sequence']
    sequence = np.array(sequence)

    if sequence.shape != (30, 132):
        return jsonify({'error': 'Invalid sequence shape, expected (30, 132)'}), 400

    action = process_sequence(sequence)

    if action:
        return jsonify({'action': action})
    else:
        return jsonify({'action': 'No action detected'})


if __name__ == '__main__':
    app.run(debug=True)
