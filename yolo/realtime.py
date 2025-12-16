import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque
from ultralytics import YOLO
import pandas as pd
import tqdm
import seaborn as sn
import torch
from PIL import Image, ImageDraw

# --- 1. CONFIGURATION ---
CLASSES = ['before', 'candy', 'computer', 'cool', 'cousin', 'drink', 'go', 'help', 'thin', 'who']
MODEL_PATH = "best_transformer_paper_spec.pth"
CONFIDENCE_THRESHOLD = 0.7 # High confidence because model is accurate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- 2. DEFINE MODEL (Must match Training EXACTLY: 6 Layers, 108 Dim) ---
class Sign2PoseTransformer(nn.Module):
    def __init__(self, input_dim=150, num_classes=10, dim_model=108, num_heads=4, num_layers=6):
        super(Sign2PoseTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, dim_model)
        self.positional_encoding = nn.Parameter(torch.randn(1, 30, dim_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True, dropout=0.4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.class_query = nn.Parameter(torch.randn(1, 1, dim_model))
        decoder_layer = nn.TransformerDecoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True, dropout=0.4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(dim_model, num_classes)

    def forward(self, src):
        if src.dim() == 2: src = src.view(src.size(0), 30, -1)
        x = self.embedding(src) + self.positional_encoding
        memory = self.encoder(x)
        batch_size = src.size(0)
        query = self.class_query.expand(batch_size, -1, -1)
        out = self.decoder(query, memory)
        out = self.fc(out.squeeze(1))
        return out

# --- 3. LOAD MODELS ---
print("Loading Models...")
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_yolo.classes = [0] # Person only

model = Sign2PoseTransformer(num_classes=len(CLASSES), input_dim=150, dim_model=108, num_layers=6).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Models Loaded Successfully!")

# --- 4. PREPROCESSING HELPER (The "Secret Sauce" logic) ---
def get_normalization_factors(landmarks):
    # Nose=0, Left Eye=2, Right Eye=5 (MediaPipe Pose indices)
    nose = np.array([landmarks[0].x, landmarks[0].y])
    left_eye = np.array([landmarks[2].x, landmarks[2].y])
    right_eye = np.array([landmarks[5].x, landmarks[5].y])
    head_width = np.linalg.norm(left_eye - right_eye)
    if head_width == 0: head_width = 0.01
    return nose, head_width

def normalize_coordinates(landmark, nose, head_width):
    norm_x = (landmark.x - nose[0]) / (head_width * 7)
    norm_y = (landmark.y - nose[1]) / (head_width * 8)
    return [norm_x, norm_y] # NO Z-Coord!

def process_landmarks(results):
    """
    Extracts, Normalizes, and Reduces Features to 150 dims (X,Y only)
    """
    if not results.pose_landmarks: return np.zeros(150)

    # 1. Get Normalization Factors
    nose, head_width = get_normalization_factors(results.pose_landmarks.landmark)
    
    # 2. Process Pose (33 points)
    pose_flat = []
    for lm in results.pose_landmarks.landmark:
        n = normalize_coordinates(lm, nose, head_width)
        pose_flat.extend(n) # Append X, Y only
        # Note: We discarded visibility and Z to match training
    
    # 3. Process Left Hand (21 points)
    lh_flat = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            n = normalize_coordinates(lm, nose, head_width)
            lh_flat.extend(n)
    else:
        lh_flat = [0] * 42 # 21*2

    # 4. Process Right Hand (21 points)
    rh_flat = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            n = normalize_coordinates(lm, nose, head_width)
            rh_flat.extend(n)
    else:
        rh_flat = [0] * 42

    # Combine: (33*2) + (21*2) + (21*2) = 66 + 42 + 42 = 150
    # Wait! 33*4 in original vs 33*2 here.
    # Training code used: Pose(132)->taken 66.
    
    # Let's ensure strict size matching to training:
    # Training 'reduce_dimensions' took:
    # Pose: 33*4 (orig) -> 33*2 (new) = 66
    # LH: 21*3 (orig) -> 21*2 (new) = 42
    # RH: 21*3 (orig) -> 21*2 (new) = 42
    # Total = 150. Perfect.
    
    return np.concatenate([pose_flat, lh_flat, rh_flat])

# --- 5. REAL-TIME LOOP ---
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)
sequence = deque(maxlen=30)
current_word = "..."
confidence_bar = 0.0

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # YOLO Crop
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_yolo = model_yolo(image_rgb)
        h, w, _ = frame.shape
        x1, y1, x2, y2 = 0, 0, w, h
        
        df_yolo = results_yolo.pandas().xyxy[0]
        if not df_yolo.empty:
            best = df_yolo.iloc[0]
            if best['confidence'] > 0.4:
                pad = 20
                x1 = max(0, int(best.xmin) - pad)
                y1 = max(0, int(best.ymin) - pad)
                x2 = min(w, int(best.xmax) + pad)
                y2 = min(h, int(best.ymax) + pad)

        process_frame = frame[y1:y2, x1:x2]
        if process_frame.size == 0: process_frame = frame

        # MediaPipe
        img_crop_rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_crop_rgb)

        # Draw box on main frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if results.pose_landmarks:
            # New Processing Logic (150 Dims)
            keypoints = process_landmarks(results)
            sequence.append(keypoints)

            if len(sequence) == 30:
                input_seq = np.array(sequence)
                input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    prediction = model(input_tensor)
                    probs = torch.softmax(prediction, dim=1)
                    conf, predicted_idx = torch.max(probs, 1)
                    
                    confidence_bar = conf.item()
                    if conf.item() > CONFIDENCE_THRESHOLD:
                        current_word = CLASSES[predicted_idx.item()]
        
        # UI Visualization
        # Background bar
        cv2.rectangle(frame, (0,0), (300, 40), (0,0,0), -1)
        # Confidence bar (Green)
        bar_width = int(confidence_bar * 300)
        cv2.rectangle(frame, (0,0), (bar_width, 40), (0, 200, 0), -1)
        # Text
        cv2.putText(frame, f"{current_word} ({int(confidence_bar*100)}%)", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Sign Language Detector (State of the Art)', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()