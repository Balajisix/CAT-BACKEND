from dotenv import load_dotenv
load_dotenv()
from app import create_app
from app.extensions import db
from app.models.models1 import VehicleLog

app = create_app()

with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(debug=True)
