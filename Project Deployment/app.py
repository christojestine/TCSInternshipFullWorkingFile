import numpy as np
from flask import Flask,request,render_template
import pickle

app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=["POST"])
def predict():
    print('step is working')
    value=[int(x) for x in request.form.values()]
    values=np.array(value)
    print(values)
    prediction=model.predict([values])
    output_value=prediction[0]
    return render_template('index.html',Prediction_output=output_value)
if __name__=='__main__':
    app.run()