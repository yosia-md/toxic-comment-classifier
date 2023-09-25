
from flask import Flask,request,render_template
import pickle
import tensorflow as tf
import keras
PATH = 'D:\KULIAH\SEMESTER 4\MACHINE LEARNING\Deployment\Toxic Classification-20230530T023239Z-001\my_h5_model.h5'
#Reading the model from JSON file
model= keras.models.load_model(PATH)
import numpy as np
from keras.preprocessing.text import Tokenizer
max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
maxlen = 200
from keras.utils import pad_sequences


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route ('/predict',methods=['POST'])
def predict():
    result = []
    text1 = request.form.get("text1")
    tokenizer.fit_on_texts(text1)
    list_tokenized_test = tokenizer.texts_to_sequences(text1)
    text_X = pad_sequences(list_tokenized_test, maxlen=maxlen)
    y_pred = model.predict(text_X)
    tf.config.run_functions_eagerly(True)
    onearray =  np.ndarray.flatten(y_pred)
    toxic = int(onearray[0]*100)
    severe_toxic = int(onearray[1]*100)
    obscene = int(onearray[2]*100)
    threat = int(onearray[3]*100)
    insult = int(onearray[4]*100)
    identity_hate = int(onearray[5]*100)
    return render_template ("index.html",toxic=toxic,severe_toxic=severe_toxic,obscene=obscene,threat=threat,insult=insult,identity_hate=identity_hate)
    
    

if __name__ == '__main__':
    app.run(debug=True)