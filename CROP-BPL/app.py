from flask import Flask, render_template, request
import pickle
import numpy as np
import requests

app = Flask(__name__)
model = pickle.load(open('model/crop_model.pkl', 'rb'))

average_yields = {
    "rice": 2700, "maize": 3000, "jute": 2300, "cotton": 1500, "mango": 8390,
    "coconut": 7000, "papaya": 40000, "orange": 12000, "apple": 10000,
    "muskmelon": 15000, "watermelon": 20000, "grapes": 18000, "banana": 30000,
    "pomegranate": 12000, "lentil": 900, "blackgram": 700, "mungbeans": 800,
    "mothbeans": 600, "pigeonpeas": 900, "kidneybeans": 1000, "chickpeas": 1000, "coffee": 800
}

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/recommend')
def recommend():
    return render_template('index.html')

@app.route('/schemes')
def schemes():
    return render_template('schemes.html')

@app.route('/history')
def history():
    base_productions = {
        "rice": 2700, "maize": 3000, "jute": 2300, "cotton": 1500, "mango": 8390,
        "coconut": 7000, "papaya": 40000, "orange": 12000, "apple": 10000,
        "muskmelon": 15000, "watermelon": 20000, "grapes": 18000, "banana": 30000,
        "pomegranate": 12000, "lentil": 900, "blackgram": 700, "mungbeans": 800,
        "mothbeans": 600, "pigeonpeas": 900, "kidneybeans": 1000, "chickpeas": 1000, "coffee": 800
    }
    return render_template('history.html', base_productions=base_productions)


@app.route('/predict', methods=['POST'])
def predict():
    N = float(request.form['nitrogen'])
    P = float(request.form['phosphorus'])
    K = float(request.form['potassium'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    land_area = float(request.form['area'])

    data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    predictions = model.predict_proba(data)[0]
    crop_names = model.classes_

    top_indices = predictions.argsort()[-3:][::-1]
    top_crops = [(crop_names[i], round(average_yields[crop_names[i]] * land_area, 2)) for i in top_indices]

    return render_template('index.html', prediction=top_crops)

if __name__ == '__main__':
    app.run(debug=True)
