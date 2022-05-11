from flask import Flask,render_template,url_for,request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

mle = pickle.load(open('mle_deployment_senti_model.pkl','rb'))
tfidf_vect = pickle.load(open('tfidf_imdb.pkl','rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():

    if request.method == 'POST':
        Reviews = request.form['Reviews']
        data = [Reviews]
        vect = tfidf_vect.transform(data).toarray()
        my_prediction = mle.predict(vect)
    return render_template('predict.html',prediction = my_prediction)



if __name__ == '__main__':
    app.run(debug=True)