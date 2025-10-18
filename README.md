
# Optivue - Eye Exercise Tracking Application

An AI-powered eye exercise application that uses MediaPipe face detection to help users perform guided eye exercises with real-time tracking and feedback.

## Features

- **Real-time Face Detection**: Uses MediaPipe's FaceDetection model for accurate face tracking
- **Guided Eye Exercises**: Multiple exercise types including focus training, eye movement patterns, and relaxation exercises
- **Visual & Audio Feedback**: Voice notifications and visual alerts when user loses focus
- **Performance Metrics**: Track your progress with detailed statistics and exercise history
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Tech Stack

**Backend:**
- Python 3.11+
- Flask (Web Framework)
- MediaPipe (Face Detection)
- OpenCV (Image Processing)
- NumPy (Array Operations)

**Frontend:**
- React 18
- TypeScript
- Tailwind CSS
- Vite (Build Tool)

## Prerequisites

Before you begin, ensure you have the following installed:
- Python 3.11 or higher
- Node.js 20 or higher
- npm or yarn package manager
- Git

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/jothi-keerthana19/Optivue.git
cd Optivue
```

### 2. Install Python Dependencies

The project uses modern Python dependency management. Install all required packages:

```bash
pip install flask flask-cors mediapipe opencv-python numpy pillow requests
```

Or if you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 3. Install Node.js Dependencies

Install the frontend dependencies:

```bash
npm install
```

## Running the Application

### Method 1: Using the Startup Script (Recommended)

The easiest way to start all servers is using the provided startup script:

```bash
python start_servers.py
```

This will start:
- **Eye Tracking Server** on port 5001
- **Enhanced Eye Tracking Server** on port 5002
- **Main Application** on port 5000 (with frontend)

### Method 2: Manual Server Startup

If you prefer to start servers individually:

#### Terminal 1 - Start Enhanced Eye Tracking Server:
```bash
python server/enhanced_eye_tracking_server.py
```

#### Terminal 2 - Start Main Application:
```bash
npm run dev
```

This will start the frontend dev server and proxy API requests to the backend.

## Accessing the Application

Once all servers are running:

1. **Open your web browser**
2. **Navigate to:** `http://localhost:5000`

The application will automatically:
- Initialize the camera (you'll need to grant camera permissions)
- Start the face detection system
- Display the exercise interface

## Project Structure

```
Optivue/
├── server/
│   ├── enhanced_eye_tracker.py          # Core face detection logic
│   ├── enhanced_eye_tracking_server.py  # Backend API server
│   ├── main_app.py                      # Frontend server & API proxy
│   └── index.ts                         # Node.js entry point
├── templates/
│   └── eye_exercises.html               # Main application page
├── client/
│   └── src/                             # React frontend components
├── start_servers.py                     # Multi-server startup script
├── package.json                         # Node.js dependencies
└── pyproject.toml                       # Python dependencies
```

## Usage Guide

### Starting an Exercise

1. Click the **"Start Camera"** button to enable face tracking
2. Grant camera permissions when prompted
3. Choose an exercise from the exercise list
4. Click **"Start Exercise"** to begin
5. Follow the on-screen instructions and visual guides

### Exercise Types

- **Focus Training**: Maintain focus on a target point
- **Eye Movement**: Follow moving patterns to strengthen eye muscles
- **20-20-20 Rule**: Look at distant objects every 20 minutes
- **Relaxation**: Blinking and palming exercises

### Notifications & Alerts

The app provides:
- **Visual alerts** when you lose focus or look away
- **Voice notifications** to guide you back to the exercise
- **Progress tracking** showing time spent and accuracy

## Troubleshooting

### Camera Not Starting
- Ensure you've granted camera permissions in your browser
- Check if another application is using the camera
- Try refreshing the page

### Face Detection Not Working
- Ensure proper lighting conditions
- Position your face clearly in front of the camera
- Check that the camera feed is visible in the preview window

### Server Connection Issues
- Verify all servers are running (ports 5000, 5001, 5002)
- Check console for error messages
- Try restarting the servers using `python start_servers.py`

### Port Already in Use
If you get a "port already in use" error:
```bash
# Kill process on port 5000 (Linux/Mac)
lsof -ti:5000 | xargs kill -9

# Kill process on port 5000 (Windows)
netstat -ano | findstr :5000
taskkill /PID <PID> /F
```

## Development

### Running in Development Mode

```bash
npm run dev
```

This starts the Vite dev server with hot module replacement.

### Building for Production

```bash
npm run build
```

Builds the frontend and backend for production deployment.

### Starting Production Server

```bash
npm start
```

## Configuration

### Environment Variables

You can configure the application using environment variables:

- `PORT`: Main application port (default: 5000)
- `TRACKING_SERVER_URL`: Eye tracking server URL (default: http://localhost:5001)
- `ENHANCED_TRACKING_SERVER_URL`: Enhanced tracking server URL (default: http://localhost:5002)

## Browser Compatibility

- Chrome 90+ (Recommended)
- Firefox 88+
- Safari 14+
- Edge 90+

**Note:** Camera access requires HTTPS in production or localhost in development.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the troubleshooting section above

## Acknowledgments

- MediaPipe for face detection technology
- OpenCV for image processing
- Flask and React communities

---

**Made with ❤️ for better eye health**
