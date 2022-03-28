from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    #print(f"Request: {request.get_json()}")
    #data = request.get_json()

    data = [float(x) for x in request.form.values()]
    input_data = [np.array(data)]

    #sl = data["sepal_length"]
    #sw = data["sepal_width"]
    #pl = data["petal_length"]
    #pw = data["petal_width"]
    #input_data = np.array([[sl, sw, pl, pw]])
    
    model_pred = model.predict(input_data)
    
    if model_pred[0] == 0:
        prediction = "Setosa"
    elif model_pred[0] == 1:
        prediction = "Versicolor"
    elif model_pred[0] == 2:
        prediction = "Virginica"

    return render_template('index.html', prediction=prediction)

if __name__ == "__main__":
    app.run()

