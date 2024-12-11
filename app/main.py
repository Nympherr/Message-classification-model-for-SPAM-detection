from flask import Flask, render_template, request, jsonify
from trained_models.trained_models import use_classifying_algorithm

app = Flask(__name__, static_url_path='/assets', static_folder='assets')

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

    # Apdoroja žinutę ir gražina "ham" arba "spam"
    result = use_classifying_algorithm(message, algorithm)
    
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
