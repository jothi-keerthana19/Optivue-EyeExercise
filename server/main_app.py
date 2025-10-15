"""
Main Web Application (Frontend Server)
Serves the HTML eye exercise page and acts as a reverse proxy
for API requests to the backend tracking server.
"""

from flask import Flask, render_template, send_from_directory, request, Response
from flask_cors import CORS
import requests


app = Flask(__name__, 
            template_folder='../templates',
            static_folder='../static')
CORS(app)

# Tracking server URLs
TRACKING_SERVER_URL = 'http://localhost:5001'
ENHANCED_TRACKING_SERVER_URL = 'http://localhost:5002'


@app.route('/')
def index():
    """Serve the eye exercises HTML page."""
    return render_template('eye_exercises.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('../static', path)


# Proxy routes to enhanced tracking server
@app.route('/api/enhanced-eye-tracking/<path:path>', methods=['GET', 'POST'])
def proxy_to_enhanced_tracking_server(path):
    """
    Reverse proxy to forward API requests to the enhanced tracking server.
    """
    try:
        url = f"{ENHANCED_TRACKING_SERVER_URL}/api/enhanced-eye-tracking/{path}"
        
        if request.method == 'GET':
            # Special handling for video feed
            if path == 'video_feed':
                resp = requests.get(url, stream=True)
                return Response(
                    resp.iter_content(chunk_size=1024),
                    content_type=resp.headers.get('Content-Type')
                )
            
            # Regular GET request
            resp = requests.get(url, params=request.args)
            return Response(
                resp.content,
                status=resp.status_code,
                headers=dict(resp.headers)
            )
        
        elif request.method == 'POST':
            # Handle file uploads
            files = None
            if request.files:
                files = {}
                for key in request.files:
                    file = request.files[key]
                    files[key] = (file.filename, file.stream, file.content_type)
            
            resp = requests.post(
                url,
                json=request.get_json() if request.is_json else None,
                data=request.form if request.form else None,
                files=files
            )
            return Response(
                resp.content,
                status=resp.status_code,
                headers=dict(resp.headers)
            )
    
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'message': 'Enhanced tracking server not available. Please ensure it is running on port 5002.'
        }, 503
    
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }, 500


# Proxy routes to tracking server
@app.route('/api/<path:path>', methods=['GET', 'POST'])
def proxy_to_tracking_server(path):
    """
    Reverse proxy to forward API requests to the tracking server.
    """
    try:
        url = f"{TRACKING_SERVER_URL}/api/{path}"
        
        if request.method == 'GET':
            # Special handling for video feed
            if path == 'video_feed':
                resp = requests.get(url, stream=True)
                return Response(
                    resp.iter_content(chunk_size=1024),
                    content_type=resp.headers.get('Content-Type')
                )
            
            # Regular GET request
            resp = requests.get(url, params=request.args)
            return Response(
                resp.content,
                status=resp.status_code,
                headers=dict(resp.headers)
            )
        
        elif request.method == 'POST':
            # Handle file uploads
            files = None
            if request.files:
                files = {}
                for key in request.files:
                    file = request.files[key]
                    files[key] = (file.filename, file.stream, file.content_type)
            
            resp = requests.post(
                url,
                json=request.get_json() if request.is_json else None,
                data=request.form if request.form else None,
                files=files
            )
            return Response(
                resp.content,
                status=resp.status_code,
                headers=dict(resp.headers)
            )
    
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'message': 'Tracking server not available. Please ensure it is running on port 5001.'
        }, 503
    
    except Exception as e:
        return {
            'success': False,
            'message': str(e)
        }, 500


if __name__ == '__main__':
    print("Main Application starting on port 5000...")
    print("Make sure the Eye Tracking Server is running on port 5001")
    app.run(host='0.0.0.0', port=5000, debug=True)
