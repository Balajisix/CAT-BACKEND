from app.extensions import db
from datetime import datetime

class Vehicle(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    license_plate = db.Column(db.String(20), unique=True, nullable=False)
    vehicle_type = db.Column(db.String(50))
    color = db.Column(db.String(30))
    owner_name = db.Column(db.String(100))
    authorized = db.Column(db.Boolean, default=False)
    added_on = db.Column(db.DateTime, default=datetime.utcnow)