from flask import Flask, render_template, request, jsonify
import joblib
import re
import scipy.sparse as sp

# ----- AVK modelis ------
model_avk = joblib.load('./trained_models/AVK/model.pkl')
vectorizer_avk = joblib.load('./trained_models/AVK/vectorizer.pkl')

punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'

def algorithm_avk(message):
    message_length = len(message)
    message_punct_count = len(re.findall(punctuation_pattern, message))
    message_vector = vectorizer_avk.transform([message])

    word_count = len(message.split())
    number_count = len(re.findall(r'\d+', message))
    standalone_number_count = len(re.findall(r'\b\d+\b', message))
    average_word_length = sum(len(word) for word in message.split()) / word_count if word_count > 0 else 0
    ratio_words_punctuation = word_count / message_punct_count if message_punct_count > 0 else 0

    vectorized_message_values = sp.hstack([
        message_vector,
        sp.csr_matrix([[message_length, message_punct_count, word_count, number_count, standalone_number_count, average_word_length, ratio_words_punctuation]])
    ])

    result = model_avk.predict(vectorized_message_values)

    return result[0]

app = Flask(__name__)

history = []

@app.route('/')
def index():
    return render_template('index.html', history=history)

@app.route('/classify', methods=['POST'])
def classify():
    global history
    data = request.json
    message = data.get('message', '')
    algorithm = data.get('algorithm', 'A')

    if algorithm == 'AVK':
        result = algorithm_avk(message)
    elif algorithm == 'LRK':
        result = 'neimplementuotas dar'
    elif algorithm == 'NBK':
        result = 'neimplementuotas dar'
    elif algorithm == 'k-nearest':
        result = 'neimplementuotas dar'
    else:
        result = 'Blogai pasirinktas algoritmas'
    
    call_details = {'message': message, 'algorithm': algorithm, 'result': result}
    history.append(call_details)
    return jsonify({'result': result})

@app.route('/inspect/<int:call_id>')
def inspect(call_id):
    global history
    if 0 <= call_id < len(history):
        return render_template('inspect_message.html', call=history[call_id])
    return "Call not found", 404

if __name__ == '__main__':
    app.run(debug=True)
