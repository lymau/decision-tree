from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def initialize_dataset(dataset):
    current_path = os.path.dirname(os.path.abspath(__file__))
    return pd.read_csv(os.path.join(current_path, dataset))

def scaling(value, column, isFirstDataset):
    scaler = MinMaxScaler()
    X = initialize_dataset('heart.csv') if isFirstDataset else initialize_dataset('heart_2.csv')
    scaler.fit(np.array(X[column]).reshape(-1,1))
    data = scaler.transform(np.array([value]).reshape(-1,1))
    return data[0][0]

def preprocessing(value):
    data = {}
    if (value['dataset'] == '1'):
        data['age'] = int(value['age'])
        data['cholesterol'] = float(value['cholesterol'])
        data['resting_bp'] = float(value['resting_bp'])
        data['max_hr'] = float(value['max_hr'])
        data['oldpeak'] = float(value['oldpeak'])

        scaler = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler_2.pkl'))
        num_data = scaler.transform([[data['age'], data['resting_bp'], data['cholesterol'], data['max_hr'], data['oldpeak']]])

        data['age'] = num_data[0][0]
        data['resting_bp'] = num_data[0][1]
        data['cholesterol'] = num_data[0][2]
        data['max_hr'] = num_data[0][3]
        data['oldpeak'] = num_data[0][4]

        data['sex_M'] = 1 if value['sex'] == 'M' else 0
        data['sex_F'] = 1 if value['sex'] == 'F' else 0
        data['exerciseangina_y'] = 1 if value['exercise_angina'] == 'Y' else 0
        data['exerciseangina_n'] = 1 if value['exercise_angina'] == 'N' else 0
        data['fasting_bs'] = int(value['fasting_bs'])

        if (value['chest_pain_type'] == 'TA'):
            data['chest_pain_type_ta'] = 1
            data['chest_pain_type_ata'] = 0
            data['chest_pain_type_nap'] = 0
            data['chest_pain_type_asy'] = 0
        elif (value['chest_pain_type'] == 'ATA'):
            data['chest_pain_type_ta'] = 0
            data['chest_pain_type_ata'] = 1
            data['chest_pain_type_nap'] = 0
            data['chest_pain_type_asy'] = 0
        elif (value['chest_pain_type'] == 'NAP'):
            data['chest_pain_type_ta'] = 0
            data['chest_pain_type_ata'] = 0
            data['chest_pain_type_nap'] = 1
            data['chest_pain_type_asy'] = 0
        elif(value['chest_pain_type'] == 'ASY'):
            data['chest_pain_type_ta'] = 0
            data['chest_pain_type_ata'] = 0
            data['chest_pain_type_nap'] = 0
            data['chest_pain_type_asy'] = 1

        if (value['resting_ecg'] == 'normal'):
            data['resting_ecg_normal'] = 1
            data['resting_ecg_st'] = 0
            data['resting_ecg_lvh'] = 0
        elif (value['resting_ecg'] == 'st'):
            data['resting_ecg_normal'] = 0
            data['resting_ecg_st'] = 1
            data['resting_ecg_lvh'] = 0
        elif (value['resting_ecg'] == 'lvh'):
            data['resting_ecg_normal'] = 0
            data['resting_ecg_st'] = 0
            data['resting_ecg_lvh'] = 1

        if (value['st_slope'] == 'up'):
            data['st_slope_up'] = 1
            data['st_slope_down'] = 0
            data['st_slope_flat'] = 0
        if (value['st_slope'] == 'down'):
            data['st_slope_up'] = 0
            data['st_slope_down'] = 1
            data['st_slope_flat'] = 0
        if (value['st_slope'] == 'flat'):
            data['st_slope_up'] = 0
            data['st_slope_down'] = 0
            data['st_slope_flat'] = 1
    
    else:
        data['sex'] = int(value['sex'])
        data['cp'] = int(value['cp'])
        data['fbs'] = int(value['fbs'])
        data['restecg'] = int(value['restecg'])
        data['exang'] = int(value['exang'])
        data['slope'] = int(value['slope'])
        data['ca'] = int(value['ca'])
        data['thal'] = int(value['thal'])

        data['age'] = int(value['age'])
        data['trestbps'] = float(value['trestbps'])
        data['chol'] = float(value['chol'])
        data['thalach'] = float(value['thalach'])
        data['oldpeak'] = float(value['oldpeak'])

        scaler = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scaler_1.pkl'))
        num_data = scaler.transform([[data['age'], data['trestbps'], data['chol'], data['thalach'], data['oldpeak']]])

        data['age'] = num_data[0][0]
        data['trestbps'] = num_data[0][1]
        data['chol'] = num_data[0][2]
        data['thalach'] = num_data[0][3]
        data['oldpeak'] = num_data[0][4]

        pca = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pca_1.pkl'))
        X_pca = pca.transform([[data['sex'], data['cp'], data['fbs'], data['restecg'], data['exang'], data['slope'], data['ca'], data['thal'], data['age'], data['trestbps'], data['chol'], data['thalach'], data['oldpeak']]])

        return X_pca
    return data

@app.route("/submit", methods=["POST"])
def submit():
    if request.method == 'POST':
        data = {}     
        if (request.form['dataset'] == '1'):    
            data['dataset'] = request.form.get('dataset')
            data['age'] = request.form.get('age')
            data['sex'] = request.form.get('sex')
            data['chest_pain_type'] = request.form.get('chest_pain_type')
            data['resting_bp'] = request.form.get('resting_bp')
            data['cholesterol'] = request.form.get('cholesterol')
            data['fasting_bs'] = request.form.get('fasting_bs')
            data['resting_ecg'] = request.form.get('resting_ecg')
            data['max_hr'] = request.form.get('max_hr')
            data['exercise_angina'] = request.form.get('exercise_angina')
            data['oldpeak'] = request.form.get('oldpeak')
            data['st_slope'] = request.form.get('st_slope')

            model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model_2.pkl'))

            new_data = preprocessing(data)

            to_predict = [[new_data['fasting_bs'], new_data['sex_M'], new_data['chest_pain_type_ata'], new_data['chest_pain_type_nap'], new_data['chest_pain_type_ta'], new_data['resting_ecg_normal'], new_data['resting_ecg_st'], new_data['exerciseangina_y'], new_data['st_slope_flat'], new_data['st_slope_up'], new_data['age'], new_data['resting_bp'], new_data['cholesterol'], new_data['max_hr'], new_data['oldpeak']]]

            result = model.predict(to_predict)

            prob = model.predict_proba(to_predict)

            prob = prob[0][0] if prob[0][0] > prob[0][1] else prob[0][1]
            
            predicted = 'Positif' if result[0] == 1 else 'Negatif'
        
        else:
            data['dataset'] = request.form.get('dataset')
            data['age'] = request.form.get('age')
            data['sex'] = request.form.get('sex')
            data['cp'] = request.form.get('cp')
            data['trestbps'] = request.form.get('trestbps')
            data['chol'] = request.form.get('chol')
            data['fbs'] = request.form.get('fbs')
            data['restecg'] = request.form.get('restecg')
            data['thalach'] = request.form.get('thalach')
            data['exang'] = request.form.get('exang')
            data['oldpeak'] = request.form.get('oldpeak')
            data['slope'] = request.form.get('slope')
            data['ca'] = request.form.get('ca')
            data['thal'] = request.form.get('thal')

            model = load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_model_1.pkl'))

            new_data = preprocessing(data)

            result = model.predict(new_data)

            prob = model.predict_proba(new_data)

            predicted = 'Positif' if result[0] == 1 else 'Negatif'

            prob = prob[0][0] if prob[0][0] > prob[0][1] else prob[0][1]

        return render_template('index.html', predicted=predicted, prob=prob)
