export interface DetectionResult {
  detected_word: string;
  confidence: number;
  bounding_box: [number, number, number, number]; // [x1, y1, x2, y2]
  keypoints: {
    pose: Keypoint[] | null;
    left_hand: Keypoint[] | null;
    right_hand: Keypoint[] | null;
  };
  sequence_length: number;
}

export interface Keypoint {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

export interface WebSocketMessage {
  type: 'frame' | 'result' | 'error';
  data?: string;
  message?: string;
}


