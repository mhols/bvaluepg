'''
# -*- coding: utf-8 -*-",
        "# Author: Toni Luhdo",
        "# Created on: 2025-03-21",
'''

from flask import Flask, render_template, request, jsonify
import numpy as np
from logic.model_utils import (
    generate_data,
    logistic_regression,
    polya_gamma_logreg,
    gibbs_sampler,
    plotly_trace_data,
    plotly_hist_data
)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    n = int(data.get('n'))
    param1 = float(data.get('param1'))
    param2 = float(data.get('param2'))

    X, y = generate_data(n, [param1, param2])
    beta_hat = logistic_regression(X, y)

    return jsonify({
        'beta_hat': beta_hat.tolist(),
        'X': X.tolist(),
        'y': y.tolist()
    })


@app.route('/polya', methods=['POST'])
def polya():
    data = request.get_json()
    X = np.array(data.get('X'))
    y = np.array(data.get('y'))

    beta_pg = polya_gamma_logreg(X, y)

    return jsonify({
        'beta_pg': beta_pg.tolist()
    })


@app.route('/gibbs', methods=['POST'])
def gibbs():
    data = request.get_json()
    X = np.array(data.get('X'))
    y = np.array(data.get('y'))
    num_samples = int(data.get('num_samples', 500))
    burn_in = int(data.get('burn_in', 100))

    samples = gibbs_sampler(X, y, num_samples=num_samples, burn_in=burn_in)
    mean_estimate = samples.mean(axis=0)

    trace_data = plotly_trace_data(samples)
    hist_data = plotly_hist_data(samples)

    return jsonify({
        'gibbs_mean_beta': mean_estimate.tolist(),
        'trace_plot_data': trace_data,
        'hist_plot_data': hist_data
    })


if __name__ == '__main__':
    app.run(debug=True)