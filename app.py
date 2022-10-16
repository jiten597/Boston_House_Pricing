import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, app, jsonify, url_for, render_template

app    = Flask(__name__)
reg    = pickle.load(open("Regression_model.pkl", 'rb'))
scaler = pickle.load(open("StandardScaler.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods = ['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = reg.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods = ['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    new_data = scaler.transform(np.array(data).reshape(1,-1))
    print(new_data)
    output = reg.predict(new_data)
    print(output[0])
    return render_template("home.html", prediction_text = f"The predicted price is: {output[0]}")

if __name__ == "__main__":
    app.run(debug = True)