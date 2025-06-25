from flask import Flask, render_template, request
import pickle
import numpy as np
import joblib
app = Flask(__name__)

# Load your trained model
model = joblib.load("model_compressed.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        holiday = int(request.form['holiday'])
        temp = float(request.form['temp'])
        rain = float(request.form['rain'])
        snow = float(request.form['snow'])
        weather = int(request.form['weather'])
        year = int(request.form['year'])
        month = int(request.form['month'])
        day = int(request.form['day'])
        hours = int(request.form['hours'])
        minutes = int(request.form['minutes'])
        seconds = int(request.form['seconds'])

        # Construct feature array
        features = np.array([[holiday, temp, rain, snow, weather, year, month, day, hours, minutes, seconds]])

        # Prediction
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction_text=f"Estimated Traffic Volume: {int(prediction)}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
