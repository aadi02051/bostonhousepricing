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
if __name__=="__main__":
    app.run(debug=True)
