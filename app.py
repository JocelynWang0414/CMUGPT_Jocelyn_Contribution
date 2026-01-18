from flask import Flask, send_from_directory, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv

load_dotenv()

# Import modules
from db_models import db

def create_app(config_object=None):
    """Create and configure the Flask application"""
    app = Flask(__name__, 
        static_folder='static',
        static_url_path=''
    )
    
    # Load configuration
    if config_object:
        app.config.from_object(config_object)
    else:
        # Load from config module
        app.config.from_object('config')
    
    # Initialize extensions
    CORS(app)
    db.init_app(app)
    
    from api import api_blueprint, init_app
    init_app(app)  # This will import the routes and register the blueprint
    
    # Serve the simple frontend at root
    @app.route('/')
    def index():
        return send_from_directory('static', 'index.html')
    
    @app.route('/userform')
    def render_userform():
        return render_template('user_form.html')

    return app

if __name__ == '__main__':
    app = create_app()
    
    if os.getenv('FLASK_ENV') == 'development':
        # Create tables in development mode if they don't exist
        with app.app_context():
            db.create_all()
    
    app.run(debug=app.config.get('DEBUG', True))