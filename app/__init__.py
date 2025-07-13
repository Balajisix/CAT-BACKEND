from flask import Flask, send_from_directory
from app.extensions import db, bcrypt, cors 
from config import Config
from flask_cors import CORS
from flask_session import Session
import os
from app.routes.image_routes import image_bp
from app.routes.vehicle_routes import vehicle_bp
from flask_migrate import Migrate


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config['SESSION_TYPE'] = 'filesystem'  
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = True
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax' 
    app.config['SESSION_COOKIE_SECURE'] = False 
    db.init_app(app)
    bcrypt.init_app(app)
    Session(app)  
    CORS(app, resources={r"/api/*": {"origins": "http://localhost:5173", "supports_credentials": True}})
    migrate = Migrate(app, db)
    from .routes.auth_routes import auth_bp
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(image_bp, url_prefix='/api/admin')
    app.register_blueprint(vehicle_bp, url_prefix="/api/admin")
    @app.route('/uploads/<filename>')
    def uploaded_file(filename):
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'uploads')
        return send_from_directory(uploads_dir, filename)

    return app
