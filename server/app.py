from flask import Flask,request
from flask_restful import Resource, Api
import time
import pickle
import pandas as pd
from flask_cors import CORS


app = Flask(__name__)
CORS(app)
api = Api(app)

class prediction(Resource):
    def get(self):
        model = pickle.load(open('pickle_model.pkl', 'rb'))
        y_pred = model.predict()
        #print("Tiempo en predicción:", end - start, "[s]")
        return str(prediction)

api.add_resource(prediction,'/prediction')

if __name__ == '__main__':
    app.run(debug=True)