// Simple launcher to start Python servers
// This just spawns the Python processes - all application logic is in Python

import { spawn } from 'child_process';

console.log('============================================================');
console.log('Eye Exercise Tracker - Launching Python Servers');
console.log('============================================================');

// Start Eye Tracking Server (port 5001)
console.log('Starting Eye Tracking Server on port 5001...');
const trackingServer = spawn('python', ['server/eye_tracking_server.py'], {
  stdio: 'inherit'
});

// Start Enhanced Eye Tracking Server (port 5002)
console.log('Starting Enhanced Eye Tracking Server on port 5002...');
const enhancedTrackingServer = spawn('python', ['server/enhanced_eye_tracking_server.py'], {
  stdio: 'inherit'
});

// Wait for tracking servers to start
setTimeout(() => {
  // Start Main Application (port 5000)
  console.log('Starting Main Application on port 5000...');
  const mainApp = spawn('python', ['server/main_app.py'], {
    stdio: 'inherit'
  });

  console.log('============================================================');
  console.log('Application started!');
  console.log('Eye Tracking Server: http://localhost:5001');
  console.log('Enhanced Eye Tracking Server: http://localhost:5002');
  console.log('Main Application: http://localhost:5000');
  console.log('============================================================');

  // Handle shutdown
  process.on('SIGTERM', () => {
    trackingServer.kill();
    enhancedTrackingServer.kill();
    mainApp.kill();
  });

  process.on('SIGINT', () => {
    trackingServer.kill();
    enhancedTrackingServer.kill();
    mainApp.kill();
  });
}, 2000);
