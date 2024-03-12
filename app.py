import pickle
from flask import Flask,request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import json

app = Flask(__name__)
model = pickle.load(open('model_boston.pkl','rb'))
scalar = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output = model.predict(new_data)
    print(output[0])

    # Convert NumPy array to list
    arr_list = output[0].tolist()

    print(arr_list[0])
    return jsonify(arr_list[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scalar.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = model.predict(final_input)[0]
    print(output)
    return render_template('home.html',prediction_text='the predicted price is {}'.format(output))

if __name__=="__main__":
    app.run(debug=True)
