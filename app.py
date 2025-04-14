from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle
import requests

from sklearn.preprocessing import LabelEncoder


app = Flask(__name__, template_folder='./templates')

# with open(r"random_forest_model.pkl", "rb") as f:
#     model = pickle.load(f)


# with open(r"scaler.pkl", "rb") as f:
#     scaler = pickle.load(f)


# with open("label_encoders.pkl", "rb") as f:
#     label_encoders = pickle.load(f)

API_URL = "https://s6sj1bgn6l.execute-api.ap-south-1.amazonaws.com/dev/predict-yield"




@app.route('/', methods=['GET', 'POST'])
def predict_coffe_yield():
    predicted_yield = None
    prediction_data = {}

    if request.method == 'GET':
        return render_template('index.html', prediction=None, prediction_data = prediction_data)
    if request.method == 'POST':
     
        district = request.form['District']
        season = request.form['Season']
        rainfall = request.form['Rainfall_mm']
        temperature = request.form['Temperature_C']
        soil_moisture = request.form['Soil_Moisture']
        pest_infestation = request.form['Pest_Infestation']
        fertilizer_usage = request.form['Fertilizer_Usage']
        try:
            json = {
               "district":district,
                "season":season,
                "rainfall":rainfall,
                "temperature":temperature,
                "soil_moisture":soil_moisture,
                "pest_infestation":pest_infestation,
                "fertilizer_usage":fertilizer_usage
            }
            import pdb
            pdb.set_trace()
            response = requests.post(API_URL, json=json)
            
            if response.status_code== 200:
                data = response.json()
                prediction_data = json
                predicted_yield = data.get('predicted_coffee_yield', '')
            else:
                prediction = f"Error occurred: {response.status_code}"
        except Exception as e:
            prediction = f"Error in loading FAST API Url: {e}"
        else:
            prediction = 'Please enter a valid description.'

    return render_template('index.html', prediction=predicted_yield, prediction_data = prediction_data)


if __name__ == '__main__':
    app.run(debug=True)
