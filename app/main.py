import json
import matplotlib.pyplot as plt
import matplotlib
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from flask import Flask, render_template, request, jsonify
from trained_models.trained_models import use_classifying_algorithm
matplotlib.use('Agg')

app = Flask(__name__, static_url_path='/assets', static_folder='assets')

history = []

@app.template_filter('b64encode')
def base64_encode(value):
    return base64.b64encode(value).decode('utf-8')

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

    json_file_path = f"trained_models/{algorithm}/metadata.json"

    with open(json_file_path, 'r') as f:
        algorithm_data = json.load(f)

    data_graph = generate_pie_chart(algorithm_data['training_data']['distribution'])
    model_graph = generate_pie_chart(algorithm_data['model_results'])

    call_details = {'message': message, 'algorithm': algorithm, 'result': result, 'algorithm_data':algorithm_data, 'data_graph':data_graph, 'model_graph':model_graph}

    history.append(call_details)
    return jsonify({'result': result})

@app.route('/inspect/<int:call_id>')
def inspect(call_id):
    global history
    if 0 <= call_id < len(history):

        call=history[call_id]

        return render_template('inspect_message.html', call=call)
    return "Call not found", 404

def generate_pie_chart(data):
    matplotlib.use('Agg')

    labels = ['Ham', 'Spam']
    sizes = [data['ham'], data['spam']]

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)

    return buf.getvalue()

if __name__ == '__main__':
    app.run(debug=True)
