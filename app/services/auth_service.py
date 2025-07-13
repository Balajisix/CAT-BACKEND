from ..models.user import User
from ..extensions import db
from ..utils.hashing import hash_password, check_password

def register_user(username, email, password):
    if User.query.filter((User.email == email) | (User.username == username)).first():
        return None, "User already exists"

    hashed_pw = hash_password(password)
    user = User(username=username, email=email, password=hashed_pw)
    db.session.add(user)
    db.session.commit()
    return user, None

def login_user(email, password):
    user = User.query.filter_by(email=email).first()
    if user and check_password(user.password, password):
        return user, None
    return None, "Invalid credentials"
