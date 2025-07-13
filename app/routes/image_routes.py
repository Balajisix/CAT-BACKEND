import os
import uuid
import random
import string
import traceback
from flask import Blueprint, request, jsonify, send_from_directory
from ultralytics import YOLO
from PIL import Image
import easyocr
import numpy as np
from app.models.models1 import VehicleLog
from app.models.vehicle import Vehicle
from app.extensions import db
from datetime import datetime
from sqlalchemy import func
from huggingface_hub import hf_hub_download

image_bp = Blueprint('image_bp', __name__)

yolo_model = hf_hub_download(
    repo_id="balaji2003/yolov8x-model",
    filename="yolov8x.pt"
)

license_model = hf_hub_download(
    repo_id="balaji2003/yolov8x-model",
    filename="license_plate_detector.pt"
)

# Load models
vehicle_model = YOLO(yolo_model)
plate_model = YOLO(license_model)

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
vehicle_data_cache = {}

def generate_asset_id():
    prefix = "ASSET"
    part1 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    part2 = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"{prefix}-{part1}-{part2}"

@image_bp.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@image_bp.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No image selected"}), 400

    file_extension = os.path.splitext(image.filename)[1]
    unique_filename = f"{uuid.uuid4().hex}{file_extension}"
    image_path = os.path.join(UPLOAD_FOLDER, unique_filename)

    try:
        print("Saving uploaded image to:", image_path)
        image.save(image_path)

        # Step 1: Detect vehicle
        print("Running vehicle detection...")
        results = vehicle_model(image_path)
        boxes = results[0].boxes.data.cpu().numpy()

        if boxes.shape[0] == 0:
            return jsonify({"error": "No vehicle detected in the image"}), 400

        class_id = int(results[0].boxes.cls[0])
        confidence = float(results[0].boxes.conf[0])
        class_map = {
            0: "Person", 1: "Bicycle", 2: "Car", 3: "Motorcycle", 
            5: "Bus", 7: "Truck"
        }
        asset_name = class_map.get(class_id, f"Unknown-{class_id}")

        # Step 2: Detect license plate
        print("Running license plate detection...")
        plate_results = plate_model(image_path)
        plate_boxes = plate_results[0].boxes.data.cpu().numpy()
        plate_number = None

        if plate_boxes.shape[0] > 0:
            xyxy = plate_results[0].boxes.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            print(f"Cropping license plate from: ({x1}, {y1}) to ({x2}, {y2})")
            pil_img = Image.open(image_path)
            plate_crop = pil_img.crop((x1, y1, x2, y2))

            # Step 3: OCR on cropped plate
            print("Performing OCR on license plate...")
            plate_crop_np = np.array(plate_crop)
            ocr_results = reader.readtext(plate_crop_np)

            if ocr_results:
                plate_number = ocr_results[0][1].replace(" ", "").upper()
                print("Detected license plate number:", plate_number)
            else:
                print("OCR failed to detect plate number.")

        # Step 4: Set asset ID
        asset_id = plate_number if plate_number else generate_asset_id()
        image_path_for_frontend = image_path.replace("\\", "/")
        image_url = f"http://localhost:5000/{image_path_for_frontend}"
        session_id = str(uuid.uuid4())

        vehicle_data_cache[session_id] = {
            "asset_id": asset_id,
            "asset_name": asset_name,
            "image_path": image_path_for_frontend
        }

        return jsonify({
            "session_id": session_id,
            "asset_id": asset_id,
            "asset_name": asset_name,
            "image_path": image_path_for_frontend,
            "image_url": image_url,
            "confidence": round(confidence, 3),
            "detected_class_id": class_id,
            "message": f"Vehicle detected: {asset_name} (confidence: {confidence:.1%})",
            "auto_filled": True,
            "next_step": "Review and edit the auto-filled data, then submit with driver name"
        }), 200

    except Exception as e:
        traceback.print_exc()
        try:
            os.remove(image_path)
        except:
            pass
        return jsonify({"error": f"Vehicle detection failed: {str(e)}"}), 500

@image_bp.route('/log-vehicle', methods=['POST'])
def log_vehicle():
    data = request.json
    session_id = data.get('session_id')
    direction = data.get('direction')  # 'inbound' or 'outbound'
    driver_name = data.get('driver_name', 'Unknown')

    if not session_id or session_id not in vehicle_data_cache:
        return jsonify({'error': 'Invalid or expired session_id'}), 400
    if direction not in ['inbound', 'outbound']:
        return jsonify({'error': 'Invalid direction'}), 400

    cached = vehicle_data_cache[session_id]
    license_plate = cached['asset_id']
    asset_name = cached['asset_name']
    image_path = cached['image_path']

    # Find vehicle in DB
    vehicle = Vehicle.query.filter_by(license_plate=license_plate).first()
    is_authorized = vehicle.authorized if vehicle else False
    vehicle_id = vehicle.id if vehicle else None

    log = VehicleLog(
        asset_id=license_plate,
        asset_name=asset_name,
        driver_name=driver_name,
        timestamp=datetime.utcnow(),
        image_path=image_path,
        license_plate=license_plate,
        direction=direction,
        is_authorized=is_authorized,
        vehicle_id=vehicle_id
    )
    db.session.add(log)
    db.session.commit()

    # Optionally, remove from cache
    del vehicle_data_cache[session_id]

    return jsonify({'message': 'Vehicle logged successfully', 'is_authorized': is_authorized})

@image_bp.route('/api/vehicle-logs', methods=['GET'])
def get_vehicle_logs():
    logs = VehicleLog.query.order_by(VehicleLog.timestamp.desc()).limit(50).all()
    return jsonify([
        {
            'license_plate': log.license_plate,
            'direction': log.direction,
            'timestamp': log.timestamp.isoformat(),
            'is_authorized': log.is_authorized,
            'image_path': log.image_path,
            'driver_name': log.driver_name,
            'asset_name': log.asset_name
        }
        for log in logs
    ])

@image_bp.route('/api/vehicle-counts', methods=['GET'])
def get_vehicle_counts():
    inbound = VehicleLog.query.filter_by(direction='inbound').count()
    outbound = VehicleLog.query.filter_by(direction='outbound').count()
    return jsonify({'inbound': inbound, 'outbound': outbound})

@image_bp.route('/api/vehicle-stats', methods=['GET'])
def get_vehicle_stats():
    period = request.args.get('period', 'day')
    if period == 'day':
        group_by = func.date(VehicleLog.timestamp)
    elif period == 'week':
        group_by = func.date_trunc('week', VehicleLog.timestamp)
    elif period == 'month':
        group_by = func.date_trunc('month', VehicleLog.timestamp)
    else:
        return jsonify({'error': 'Invalid period'}), 400

    results = db.session.query(
        group_by.label('period'),
        VehicleLog.direction,
        func.count().label('count')
    ).group_by(group_by, VehicleLog.direction).order_by(group_by).all()

    stats = {}
    for row in results:
        key = str(row.period)
        if key not in stats:
            stats[key] = {'inbound': 0, 'outbound': 0}
        stats[key][row.direction] = row.count

    return jsonify(stats)
