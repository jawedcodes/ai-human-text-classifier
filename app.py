from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle

# Load the trained model
model = load_model('model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Define maximum sequence length (must match training)
MAX_SEQUENCE_LENGTH = 50  # adjust if your model used a different length

# Create Flask app
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    # Preprocess input
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Predict
    pred = model.predict(padded)[0][0]
    label = "Human-Written" if pred > 0.5 else "AI-Written"

    return render_template('result.html', result=label)

if __name__ == '__main__':
    app.run(debug=True)
