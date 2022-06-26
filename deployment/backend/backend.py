from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

data = {
    'symbol': 'ABCD',
    'n_close': [1234.5678],
    'n_days': [40],
    'text': 'Halo teks abc'
}

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

@app.route("/")
def hello_world():
    return jsonify(data)

@app.route("/price", methods=['GET', 'POST'])
def data_inference():
    if request.method == 'POST':
        data = request.json

        symbol = data['symbol']
        n_days = data['n_days'][0]
        n_close_list = data['n_close']

        #Price Prediction
        model_name = {
            'gold' : 'ts_gold.hdf5',
            'dxy' : 'ts_dxy.hdf5',
            'snp500' : 'ts_snp500.hdf5',
            'oil' : 'ts_crudeoil.hdf5',
            'jkse' : 'ts_jkse.hdf5'
        }

        try:
            model = tf.keras.models.load_model(
                'model/' + model_name[
                    'gold' if symbol=='GC=F' \
                        else 'dxy' if symbol=='DX-Y.NYB' \
                            else 'snp500' if symbol=='^GSPC' \
                                else 'oil' if symbol=='CL=F' \
                                    else 'jkse' if symbol=='^JKSE' \
                                        else 'nothing'
                ]
            )
            pred_inf = model.predict(np.array(n_close_list).reshape(1, n_days, 1)) 
            pred_inf = pred_inf.tolist()
        except:
            pred_inf = 'Error! Symbol out of database'


        # Headline Prediction
        text = data['teks']
        text_series = pd.Series(text)
        input_tokenized = tokenizer.texts_to_sequences(text_series)
        input_pad = pad_sequences(input_tokenized, maxlen=100, padding='post')

        model_nlp2 = tf.keras.models.load_model('model_glove_conv1d_gap')

        predict_proba = model_nlp2.predict(input_pad)
        nlp_result = predict_proba.argmax(axis=1).tolist()

        response = {
            'code':200, 
            'status':'OK',
            'prediction': pred_inf, 
            'nlp_pred' : nlp_result
        }

        return jsonify(response)
    return "Silahkan gunakan method post untuk mengakses model"

# app.run(debug=True)
#------------------------------------------------------------------------------------------------------------