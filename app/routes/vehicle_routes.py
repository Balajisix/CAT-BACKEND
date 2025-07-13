from flask import Blueprint, request, jsonify
from app.models.models1 import VehicleLog
from app.models.vehicle import Vehicle
from app.extensions import db
import os
from datetime import datetime, timedelta
from ultralytics import YOLO
from PIL import Image
import easyocr
import numpy as np
import re
from huggingface_hub import hf_hub_download

vehicle_bp = Blueprint('vehicle_bp', __name__)

yolo_model = hf_hub_download(
    repo_id="balaji2003/yolov8x-model",
    filename="yolov8x.pt"
)

license_model = hf_hub_download(
    repo_id="balaji2003/yolov8x-model",
    filename="license_plate_detector.pt"
)

# Load models for detection
vehicle_model = YOLO(yolo_model)
plate_model = YOLO(license_model)
reader = easyocr.Reader(['en'], gpu=False)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def detect_plate_from_image(image_path):
    plate_results = plate_model(image_path)
    plate_boxes = plate_results[0].boxes.data.cpu().numpy()
    plate_number = None
    if plate_boxes.shape[0] > 0:
        xyxy = plate_results[0].boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        pil_img = Image.open(image_path)
        plate_crop = pil_img.crop((x1, y1, x2, y2))
        plate_crop_np = np.array(plate_crop)
        ocr_results = reader.readtext(plate_crop_np)
        if ocr_results:
            plate_number = ocr_results[0][1].replace(" ", "").upper()
    return plate_number

def clean_plate(plate):
    return re.sub(r'[^A-Z0-9]', '', plate.upper())

# Vehicle Registration Preview
@vehicle_bp.route('/register-vehicle/preview', methods=['POST'])
def register_vehicle_preview():
    if 'image' not in request.files:
        return jsonify({'error': 'Image is required'}), 400
    image = request.files['image']
    if image.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    file_extension = os.path.splitext(image.filename)[1]
    unique_filename = f"register_preview_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}{file_extension}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    image.save(image_path)
    
    try:
        # Detect vehicle type using YOLO
        results = vehicle_model(image_path)
        boxes = results[0].boxes.data.cpu().numpy()
        vehicle_type = None
        confidence = 0.0
        
        if boxes.shape[0] > 0:
            class_id = int(results[0].boxes.cls[0])
            confidence = float(results[0].boxes.conf[0])
            class_map = {
                0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 
                5: "Bus", 7: "Truck"
            }
            vehicle_type = class_map.get(class_id, f"Unknown-{class_id}")
        else:
            vehicle_type = "Unknown"
        
        # Detect license plate
        plate_number = detect_plate_from_image(image_path)
        license_plate = clean_plate(plate_number) if plate_number else None
        
        return jsonify({
            'vehicle_type': vehicle_type,
            'license_plate': license_plate,
            'image_path': image_path,
            'confidence': round(confidence, 3),
            'detected_class_id': int(results[0].boxes.cls[0]) if boxes.shape[0] > 0 else None
        }), 200
    except Exception as e:
        print(f"Error in register preview: {str(e)}")
        return jsonify({'error': f'Vehicle detection failed: {str(e)}'}), 500

# Vehicle Registration
@vehicle_bp.route('/register-vehicle', methods=['POST'])
def register_vehicle():
    data = request.get_json()
    license_plate = data.get('license_plate')
    vehicle_type = data.get('vehicle_type')
    color = data.get('color', 'Unknown')
    owner_name = data.get('owner_name', 'Unknown')

    if not license_plate or not vehicle_type:
        return jsonify({'error': 'License plate and vehicle type are required'}), 400

    license_plate = clean_plate(license_plate)

    # Check if already registered
    existing = Vehicle.query.filter_by(license_plate=license_plate).first()
    if existing:
        return jsonify({
            'error': 'Vehicle already registered',
            'license_plate': license_plate
        }), 409

    try:
        vehicle = Vehicle(
            license_plate=license_plate,
            vehicle_type=vehicle_type,
            color=color,
            owner_name=owner_name,
            authorized=True
        )
        db.session.add(vehicle)
        db.session.commit()
        return jsonify({
            'message': 'Vehicle registered and authorized successfully',
            'license_plate': license_plate,
            'vehicle_type': vehicle_type,
            'authorized': True
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to register vehicle: {str(e)}'}), 500

# Vehicle Check (Inbound/Outbound Detection)
@vehicle_bp.route('/check-vehicle', methods=['POST'])
def check_vehicle():
    if 'image' not in request.files or 'direction' not in request.form:
        return jsonify({'error': 'Image and direction are required'}), 400
    
    image = request.files['image']
    direction = request.form['direction']
    
    if image.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    file_extension = os.path.splitext(image.filename)[1]
    unique_filename = f"check_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}{file_extension}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)
    image.save(image_path)
    
    try:
        # Detect license plate from image
        plate_number = detect_plate_from_image(image_path)
        print("[CHECK] Detected plate (raw):", plate_number)
        
        if not plate_number:
            return jsonify({'error': 'No license plate detected'}), 400
        
        # Clean and normalize detected plate
        plate_number = clean_plate(plate_number)
        print("[CHECK] Detected plate (normalized):", plate_number)
        
        # Check if vehicle exists in database
        vehicle = Vehicle.query.filter_by(license_plate=plate_number).first()
        is_authorized = vehicle.authorized if vehicle else False
        
        # Determine message based on authorization status
        if is_authorized:
            message = '✅ Authorized Vehicle'
            status = 'authorized'
        else:
            message = '❌ Unauthorized Vehicle Detected'
            status = 'unauthorized'
        
        print(f"[CHECK] is_authorized: {is_authorized}, message: {message}")
        
        # Log the vehicle check
        log = VehicleLog(
            asset_id=plate_number,
            asset_name=vehicle.vehicle_type if vehicle else 'Unknown',
            driver_name='Gate Check',
            timestamp=datetime.utcnow(),
            image_path=image_path,
            license_plate=plate_number,
            direction=direction,
            is_authorized=is_authorized,
            vehicle_id=vehicle.id if vehicle else None
        )
        db.session.add(log)
        db.session.commit()
        
        return jsonify({
            'license_plate': plate_number,
            'is_authorized': is_authorized,
            'message': message,
            'status': status,
            'vehicle_type': vehicle.vehicle_type if vehicle else 'Unknown',
            'direction': direction
        }), 200
        
    except Exception as e:
        print(f"Error in vehicle check: {str(e)}")
        return jsonify({'error': f'Vehicle check failed: {str(e)}'}), 500

# Get all authorized vehicles
@vehicle_bp.route('/authorized-vehicles', methods=['GET'])
def get_authorized_vehicles():
    """Get all authorized vehicles from the database"""
    try:
        vehicles = Vehicle.query.filter_by(authorized=True).all()
        vehicles_data = []
        
        for vehicle in vehicles:
            vehicles_data.append({
                'id': vehicle.id,
                'license_plate': vehicle.license_plate,
                'vehicle_type': vehicle.vehicle_type,
                'color': vehicle.color,
                'owner_name': vehicle.owner_name,
                'authorized': vehicle.authorized,
                'added_on': vehicle.added_on.isoformat() if vehicle.added_on else None
            })
        
        return jsonify({
            'status': 'success',
            'count': len(vehicles_data),
            'vehicles': vehicles_data
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrieve authorized vehicles: {str(e)}'
        }), 500

# Get vehicle statistics for dashboard
@vehicle_bp.route('/vehicle-stats', methods=['GET'])
def get_vehicle_stats():
    """Get vehicle statistics for dashboard charts"""
    try:
        # Total authorized vehicles
        total_authorized = Vehicle.query.filter_by(authorized=True).count()
        
        # Vehicle types distribution
        vehicle_types = db.session.query(
            Vehicle.vehicle_type, 
            db.func.count(Vehicle.id)
        ).filter_by(authorized=True).group_by(Vehicle.vehicle_type).all()
        
        # Recent vehicle logs (last 24 hours)
        yesterday = datetime.utcnow() - timedelta(days=1)
        recent_logs = VehicleLog.query.filter(
            VehicleLog.timestamp >= yesterday
        ).count()
        
        # Authorized vs unauthorized entries
        authorized_entries = VehicleLog.query.filter_by(is_authorized=True).count()
        unauthorized_entries = VehicleLog.query.filter_by(is_authorized=False).count()
        
        # Inbound vs Outbound statistics
        inbound_count = VehicleLog.query.filter_by(direction='inbound').count()
        outbound_count = VehicleLog.query.filter_by(direction='outbound').count()
        
        return jsonify({
            'status': 'success',
            'stats': {
                'total_authorized_vehicles': total_authorized,
                'vehicle_types_distribution': dict(vehicle_types),
                'recent_entries_24h': recent_logs,
                'authorized_entries': authorized_entries,
                'unauthorized_entries': unauthorized_entries,
                'inbound_count': inbound_count,
                'outbound_count': outbound_count
            }
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrieve vehicle statistics: {str(e)}'
        }), 500

# Get recent vehicle movements (inbound/outbound)
@vehicle_bp.route('/recent-movements', methods=['GET'])
def get_recent_movements():
    """Get recent vehicle inbound/outbound movements"""
    try:
        # Get last 20 vehicle movements
        recent_movements = VehicleLog.query.filter(
            VehicleLog.direction.in_(['inbound', 'outbound'])
        ).order_by(VehicleLog.timestamp.desc()).limit(20).all()
        
        movements_data = []
        for movement in recent_movements:
            movements_data.append({
                'id': movement.id,
                'license_plate': movement.license_plate,
                'vehicle_type': movement.asset_name,
                'direction': movement.direction,
                'is_authorized': movement.is_authorized,
                'timestamp': movement.timestamp.isoformat(),
                'driver_name': movement.driver_name,
                'image_path': movement.image_path
            })
        
        return jsonify({
            'status': 'success',
            'movements': movements_data
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrieve recent movements: {str(e)}'
        }), 500

# Get vehicle movement data for different time periods
@vehicle_bp.route('/vehicle-movements/<period>', methods=['GET'])
def get_vehicle_movements(period):
    """Get vehicle movement data for different time periods"""
    try:
        now = datetime.utcnow()
        
        if period == 'today':
            # Today's data by hour
            start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = start_date + timedelta(days=1)
            
            movements = []
            for hour in range(24):
                hour_start = start_date + timedelta(hours=hour)
                hour_end = hour_start + timedelta(hours=1)
                
                inbound_count = VehicleLog.query.filter(
                    VehicleLog.direction == 'inbound',
                    VehicleLog.timestamp >= hour_start,
                    VehicleLog.timestamp < hour_end
                ).count()
                
                outbound_count = VehicleLog.query.filter(
                    VehicleLog.direction == 'outbound',
                    VehicleLog.timestamp >= hour_start,
                    VehicleLog.timestamp < hour_end
                ).count()
                
                movements.append({
                    'label': f'{hour:02d}:00',
                    'inbound': inbound_count,
                    'outbound': outbound_count
                })
                
        elif period == '7days':
            # Last 7 days data
            movements = []
            for i in range(7):
                date = now - timedelta(days=i)
                start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
                end_of_day = start_of_day + timedelta(days=1)
                
                inbound_count = VehicleLog.query.filter(
                    VehicleLog.direction == 'inbound',
                    VehicleLog.timestamp >= start_of_day,
                    VehicleLog.timestamp < end_of_day
                ).count()
                
                outbound_count = VehicleLog.query.filter(
                    VehicleLog.direction == 'outbound',
                    VehicleLog.timestamp >= start_of_day,
                    VehicleLog.timestamp < end_of_day
                ).count()
                
                movements.append({
                    'label': start_of_day.strftime('%a'),
                    'date': start_of_day.strftime('%Y-%m-%d'),
                    'inbound': inbound_count,
                    'outbound': outbound_count
                })
            movements.reverse()
            
        elif period == 'monthly':
            # Last 12 months data
            movements = []
            for i in range(12):
                date = now - timedelta(days=30*i)
                start_of_month = date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                if i == 0:
                    end_of_month = now
                else:
                    next_month = start_of_month + timedelta(days=32)
                    end_of_month = next_month.replace(day=1) - timedelta(days=1)
                
                inbound_count = VehicleLog.query.filter(
                    VehicleLog.direction == 'inbound',
                    VehicleLog.timestamp >= start_of_month,
                    VehicleLog.timestamp <= end_of_month
                ).count()
                
                outbound_count = VehicleLog.query.filter(
                    VehicleLog.direction == 'outbound',
                    VehicleLog.timestamp >= start_of_month,
                    VehicleLog.timestamp <= end_of_month
                ).count()
                
                movements.append({
                    'label': start_of_month.strftime('%b %Y'),
                    'date': start_of_month.strftime('%Y-%m'),
                    'inbound': inbound_count,
                    'outbound': outbound_count
                })
            movements.reverse()
            
        elif period == 'yearly':
            # Last 5 years data
            movements = []
            for i in range(5):
                year = now.year - i
                start_of_year = datetime(year, 1, 1)
                end_of_year = datetime(year, 12, 31, 23, 59, 59)
                
                inbound_count = VehicleLog.query.filter(
                    VehicleLog.direction == 'inbound',
                    VehicleLog.timestamp >= start_of_year,
                    VehicleLog.timestamp <= end_of_year
                ).count()
                
                outbound_count = VehicleLog.query.filter(
                    VehicleLog.direction == 'outbound',
                    VehicleLog.timestamp >= start_of_year,
                    VehicleLog.timestamp <= end_of_year
                ).count()
                
                movements.append({
                    'label': str(year),
                    'date': str(year),
                    'inbound': inbound_count,
                    'outbound': outbound_count
                })
            movements.reverse()
            
        else:
            return jsonify({'error': 'Invalid period. Use: today, 7days, monthly, yearly'}), 400
        
        return jsonify({
            'status': 'success',
            'period': period,
            'movements': movements
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Failed to retrieve movement data: {str(e)}'
        }), 500
