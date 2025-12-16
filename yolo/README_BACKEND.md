# Sign Language Translation Backend

Python WebSocket server for real-time sign language translation using YOLO, MediaPipe, and Transformer models.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the model file `best_transformer_paper_spec.pth` in the current directory.

3. Start the server:
```bash
python backend_server.py
```

The server will start on `http://localhost:5000` and wait for WebSocket connections.

## API

### WebSocket Events

**Client → Server:**
- `frame`: Send a video frame (base64 encoded image)
  ```json
  {
    "data": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }
  ```

**Server → Client:**
- `result`: Detection results
  ```json
  {
    "detected_word": "computer",
    "confidence": 0.95,
    "bounding_box": [100, 50, 500, 400],
    "keypoints": {
      "pose": [...],
      "left_hand": [...],
      "right_hand": [...]
    },
    "sequence_length": 30
  }
  ```

- `error`: Error message
  ```json
  {
    "message": "Error processing frame"
  }
  ```

## Model Details

- **YOLO**: Person detection and bounding box extraction
- **MediaPipe**: Pose and hand landmark extraction
- **Transformer Model**: Sign language classification
  - Classes: ['before', 'candy', 'computer', 'cool', 'cousin', 'drink', 'go', 'help', 'thin', 'who']
  - Confidence threshold: 0.7
  - Sequence length: 30 frames

## Notes

- Each client maintains its own sequence buffer (30 frames)
- Processing requires 30 frames before making predictions
- Frames should be sent at approximately 10-15 fps for optimal performance


