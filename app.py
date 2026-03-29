from flask import Flask,request,jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_marshmallow import Marshmallow
from decouple import config
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

#### CONFIGURACION DE SQLALCHEMY ####
app.app_context().push()
app.config['SQLALCHEMY_DATABASE_URI'] = config('DATABASE_URL')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

### CREAMOS UNA CLASE QUE VA A CONVERTIRSE EN UNA TABLA SQL

class Insurance(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    age = db.Column(db.Integer,nullable=False)
    price = db.Column(db.Double,nullable=True)
    
    def __init__(self,age):
        self.age = age
        
### CREAMOS UN ESQUEMA PAARA SERIALIZAR LOS DATOS
ma = Marshmallow(app)
class InsuranceSchema(ma.Schema):
    id = ma.Integer()
    age = ma.Integer()
    price = ma.Float()
    
## REGISTRAMOS LA TABLA EN LA BASE DE DATOS
db.create_all()
print('Tablas en base de datos creadas')


##### HOUSING ML ################
import joblib
import numpy as np
import sklearn

model = joblib.load('./model/model.pkl')
sc_x = joblib.load('./model/scaler_x.pkl')
sc_y = joblib.load('./model/scaler_y.pkl')

def predict_price(age):
    age_sc = sc_x.transform(np.array([[age]]))
    prediction = model.predict(age_sc)
    prediction_sc = sc_y.inverse_transform(prediction)
    price = round(float(prediction_sc[0][0]),2)
    return price


@app.route('/')
def index():
    context = {
        'title':'TRABAJO FINAL MOULO 8',
        'message':'AUTOR : '
    }
    return jsonify(context)

@app.route('/insurance_price',methods=['POST'])
def insurance_price():
    age = request.json['age']
    price = predict_price(age)
    context = {
        'message':'precio predicho',
        'edad': age,
        'prima seguro': price
    }
    return jsonify(context)

@app.route('/insurance',methods=['POST'])
def set_data():
    age = request.json['age']
    price = predict_price(age)
    
    new_data = Insurance(age)
    new_data.price = price
    db.session.add(new_data)
    db.session.commit()
    
    data_schema = InsuranceSchema()
    context = data_schema.dump(new_data)
    
    return jsonify(context)

if __name__ == '__main__':
    app.run(debug=True)