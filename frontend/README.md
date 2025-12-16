# Sign Language Translation Frontend

React frontend application for real-time sign language translation using YOLO, MediaPipe, and Transformer models.

## Features

- Real-time webcam video capture
- WebSocket communication with Python backend
- Live detection results display (word, confidence score)
- Bounding box visualization
- Pose and hand keypoints overlay
- Sequence buffer progress indicator

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

The application will be available at `http://localhost:3000`

## Backend Requirements

Make sure the Python backend server is running on `http://localhost:5000`.

To start the backend server:
```bash
cd ../yolo
python backend_server.py
```

## Usage

1. Allow camera permissions when prompted
2. The application will automatically connect to the backend server
3. Sign language gestures will be detected and displayed in real-time
4. The system requires 30 frames (approximately 2 seconds at 15 fps) before making predictions

## Technology Stack

- React 18
- TypeScript
- Vite
- Socket.IO Client
- HTML5 Canvas API
- MediaDevices API


