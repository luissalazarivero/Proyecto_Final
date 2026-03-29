from flask import Flask, request, jsonify
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

### MODELOS ###

class Insurance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Double, nullable=True)

    def __init__(self, age):
        self.age = age

class Housing(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    rooms = db.Column(db.Integer, nullable=False)
    price = db.Column(db.Double, nullable=True)

    def __init__(self, rooms):
        self.rooms = rooms

### ESQUEMAS ###

ma = Marshmallow(app)

class InsuranceSchema(ma.Schema):
    id = ma.Integer()
    age = ma.Integer()
    price = ma.Float()

class HousingSchema(ma.Schema):
    id = ma.Integer()
    rooms = ma.Integer()
    price = ma.Float()

## CREAR TABLAS ##
db.create_all()
print('Tablas en base de datos creadas')

##### ML INSURANCE ################
import joblib
import numpy as np
import sklearn

insurance_model = joblib.load('./model/model.pkl')
insurance_sc_x = joblib.load('./model/scaler_x.pkl')
insurance_sc_y = joblib.load('./model/scaler_y.pkl')

def predict_insurance_price(age):
    age_sc = insurance_sc_x.transform(np.array([[age]]))
    prediction = insurance_model.predict(age_sc)
    prediction_sc = insurance_sc_y.inverse_transform(prediction)
    price = round(float(prediction_sc[0][0]), 2)
    return price

##### ML HOUSING ################
# housing_model = joblib.load('./model_housing/model.pkl')
# housing_sc_x = joblib.load('./model_housing/scaler_x.pkl')
# housing_sc_y = joblib.load('./model_housing/scaler_y.pkl')

def predict_housing_price(rooms):
    rooms_sc = housing_sc_x.transform(np.array([[rooms]]))
    prediction = housing_model.predict(rooms_sc)
    prediction_sc = housing_sc_y.inverse_transform(prediction) * 1000
    price = round(float(prediction_sc[0][0]), 2)
    return price

### RUTAS GENERALES ###

@app.route('/')
def index():
    context = {
        'title': 'FLASK API VERSION 1.0',
        'message': 'Bienvenido a mi API'
    }
    return jsonify(context)

### RUTAS INSURANCE ###

@app.route('/insurance_price', methods=['POST'])
def insurance_price():
    age = request.json['age']
    price = predict_insurance_price(age)
    context = {
        'message': 'precio predicho',
        'edad': age,
        'prima seguro': price
    }
    return jsonify(context)

@app.route('/insurance', methods=['POST'])
def set_insurance():
    age = request.json['age']
    price = predict_insurance_price(age)

    new_data = Insurance(age)
    new_data.price = price
    db.session.add(new_data)
    db.session.commit()

    data_schema = InsuranceSchema()
    return jsonify(data_schema.dump(new_data))

@app.route('/insurance', methods=['GET'])
def get_insurance():
    data = Insurance.query.all()
    data_schema = InsuranceSchema(many=True)
    return jsonify(data_schema.dump(data))

@app.route('/insurance/<int:id>', methods=['GET'])
def get_insurance_by_id(id):
    data = Insurance.query.get(id)
    data_schema = InsuranceSchema()
    return jsonify(data_schema.dump(data)), 200 if data else 404

@app.route('/insurance/<int:id>', methods=['PUT'])
def update_insurance(id):
    data = Insurance.query.get(id)
    if not data:
        return jsonify({'message': 'Registro no encontrado'}), 404

    age = request.json['age']
    price = predict_insurance_price(age)

    data.age = age
    data.price = price
    db.session.commit()

    data_schema = InsuranceSchema()
    return jsonify(data_schema.dump(data)), 200

@app.route('/insurance/<int:id>', methods=['DELETE'])
def delete_insurance(id):
    data = Insurance.query.get(id)
    if not data:
        return jsonify({'message': 'Registro no encontrado'}), 404

    db.session.delete(data)
    db.session.commit()
    return jsonify({'message': 'Registro eliminado correctamente'}), 200

### RUTAS HOUSING ###

@app.route('/housing_price', methods=['POST'])
def housing_price():
    rooms = request.json['rooms']
    price = predict_housing_price(rooms)
    context = {
        'message': 'precio predicho',
        'habitaciones': rooms,
        'precio': price
    }
    return jsonify(context)

@app.route('/housing', methods=['POST'])
def set_housing():
    rooms = request.json['rooms']
    price = predict_housing_price(rooms)

    new_housing = Housing(rooms)
    new_housing.price = price
    db.session.add(new_housing)
    db.session.commit()

    data_schema = HousingSchema()
    return jsonify(data_schema.dump(new_housing))

@app.route('/housing', methods=['GET'])
def get_housing():
    data = Housing.query.all()
    data_schema = HousingSchema(many=True)
    return jsonify(data_schema.dump(data))

@app.route('/housing/<int:id>', methods=['GET'])
def get_housing_by_id(id):
    data = Housing.query.get(id)
    data_schema = HousingSchema()
    return jsonify(data_schema.dump(data)), 200 if data else 404

@app.route('/housing/<int:id>', methods=['PUT'])
def update_housing(id):
    data = Housing.query.get(id)
    if not data:
        return jsonify({'message': 'Registro no encontrado'}), 404

    rooms = request.json['rooms']
    price = predict_housing_price(rooms)

    data.rooms = rooms
    data.price = price
    db.session.commit()

    data_schema = HousingSchema()
    return jsonify(data_schema.dump(data)), 200

@app.route('/housing/<int:id>', methods=['DELETE'])
def delete_housing(id):
    data = Housing.query.get(id)
    if not data:
        return jsonify({'message': 'Registro no encontrado'}), 404

    db.session.delete(data)
    db.session.commit()
    return jsonify({'message': 'Registro eliminado correctamente'}), 200

if __name__ == '__main__':
    app.run(debug=True)