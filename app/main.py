from flask import Flask, render_template, request, jsonify
import joblib
import re
import scipy.sparse as sp

model1 = joblib.load('./models/1/model.pkl')
vectorizer1 = joblib.load('./models/1/vectorizer.pkl')

punctuation_pattern = r'[!"#$%&\'()*+,-./:;<=>?@[\\\]^_`{|}~]'

def algorithm_1(message):
    message_length = len(message)
    message_punct_count = len(re.findall(punctuation_pattern, message))
    message_vector = vectorizer1.transform([message])
    vectorized_message_values = sp.hstack([
        message_vector,
        sp.csr_matrix([[message_length, message_punct_count]])
    ])
    result = model1.predict(vectorized_message_values)
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

    if algorithm == '1':
        result = algorithm_1(message)
    elif algorithm == 'B':
        result = 'ham (Algorithm B)'
    elif algorithm == 'C':
        result = 'ham (Algorithm C)'
    else:
        result = 'Invalid Algorithm'
    
    call_details = {'message': message, 'algorithm': algorithm, 'result': result}
    history.append(call_details)
    return jsonify({'result': result})

@app.route('/inspect/<int:call_id>')
def inspect(call_id):
    global history
    if 0 <= call_id < len(history):
        return render_template('inspect.html', call=history[call_id])
    return "Call not found", 404

if __name__ == '__main__':
    app.run(debug=True)
