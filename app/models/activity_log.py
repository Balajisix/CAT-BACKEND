from app.extensions import db
from datetime import datetime

class ActivityLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicle.id'), nullable=False)
    direction = db.Column(db.String(10))  # inbound / outbound
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(200))
    vehicle = db.relationship('Vehicle', backref=db.backref('activities', lazy=True))
