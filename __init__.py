from flask import Blueprint

# Create the blueprint
api_blueprint = Blueprint('api', __name__)

# Import routes inside a function to avoid circular imports
def init_app(app):
    # Import routes AFTER creating the blueprint
    from . import chat, routes, recommendations
    app.register_blueprint(api_blueprint, url_prefix='/api')