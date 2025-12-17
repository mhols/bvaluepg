from flask import Flask, render_template, request, jsonify
import numpy as np
import plotly.graph_objs as go
import plotly.utils
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    lambda_rate = float(request.form['lambda_rate'])
    T = float(request.form['T'])
    method = request.form['method']

    if method == 'approach1':
        N = np.random.poisson(lambda_rate * T)
        samples = np.sort(np.random.uniform(0, T, N))
    elif method == 'approach2':
        t = 0
        samples = []
        while t < T:
            t += np.random.exponential(1 / lambda_rate)
            if t < T:
                samples.append(t)
        samples = np.array(samples)

    # erstelle plot
    fig = go.Figure()

    # Histogram for distribution of events
    fig.add_trace(go.Histogram(x=samples, nbinsx=30, name='Event Distribution', opacity=0.6))

    # Scatter plot fuer actual event times
    fig.add_trace(go.Scatter(x=samples, y=np.zeros_like(samples), mode='markers', marker=dict(size=10), name='Event Times'))

    fig.update_layout(
        title='Poisson Process Samples and Distribution',
        xaxis_title='Time',
        yaxis_title='Count',
        yaxis=dict(showticklabels=False),
        bargap=0.1,
        legend=dict(orientation='h')
    )
    
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return jsonify(graphJSON)

if __name__ == '__main__':
    app.run(debug=True)
