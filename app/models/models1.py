from ..extensions import db
from datetime import datetime

class VehicleLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    asset_id = db.Column(db.String(50), nullable=False)
    asset_name = db.Column(db.String(50), nullable=False)
    driver_name = db.Column(db.String(100), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(200), nullable=False)
    # New fields for enhanced logging
    license_plate = db.Column(db.String(20), nullable=True)
    direction = db.Column(db.String(10), nullable=True)  # 'inbound' or 'outbound'
    is_authorized = db.Column(db.Boolean, nullable=True)
    vehicle_id = db.Column(db.Integer, db.ForeignKey('vehicle.id'), nullable=True)
