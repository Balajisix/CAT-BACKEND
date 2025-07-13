from flask import Blueprint, request, jsonify
from ..services.auth_service import register_user, login_user

auth_bp = Blueprint('auth_bp', __name__)

@auth_bp.route('/register', methods=['POST'])
def signup():
    data = request.get_json()
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')

    if not all([username, email, password]):
        return jsonify({'message': 'All fields are required'}), 400

    user, error = register_user(username, email, password)
    if error:
        return jsonify({'message': error}), 409

    return jsonify({'message': 'User registered successfully'}), 201

@auth_bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')

    if not all([email, password]):
        return jsonify({'message': 'Email and password required'}), 400

    user, error = login_user(email, password)
    if error:
        return jsonify({'message': error}), 401

    return jsonify({'message': 'Login successful', 'username': user.username}), 200
