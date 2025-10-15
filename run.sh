#!/bin/bash

# Eye Exercise Tracker - Server Startup Script
echo "============================================================"
echo "Starting Eye Exercise Tracker Servers"
echo "============================================================"

# Start Eye Tracking Server in background
echo "Starting Eye Tracking Server on port 5001..."
python server/eye_tracking_server.py > /dev/null 2>&1 &
TRACKING_PID=$!

# Wait for tracking server to initialize
sleep 2

# Start Main Application
echo "Starting Main Application on port 5000..."
python server/main_app.py &
MAIN_PID=$!

# Wait for main app to initialize
sleep 2

echo "============================================================"
echo "Servers started successfully!"
echo "Eye Tracking Server (PID: $TRACKING_PID) - Port 5001"
echo "Main Application (PID: $MAIN_PID) - Port 5000"
echo "============================================================"
echo "Application URL: http://localhost:5000"
echo "============================================================"

# Keep script running
wait $MAIN_PID
