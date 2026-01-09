import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from collections import deque
from queue import Queue, Empty
import base64
from flask import Flask, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import json
import io
import gc
import sys

CLASSES = ['before', 'candy', 'computer', 'cool', 'cousin', 'drink', 'go', 'help', 'thin', 'who']
MODEL_PATH = "best_transformer_paper_spec.pth"
CONFIDENCE_THRESHOLD = 0.7

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

print("Loading Models...")
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model_yolo.classes = [0]

model = Sign2PoseTransformer(num_classes=len(CLASSES), input_dim=150, dim_model=108, num_layers=6).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print("Models Loaded Successfully!")

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    static_image_mode=False,
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    model_complexity=1
)

client_sequences = {}

processing_locks = {}
frame_queues = {}

def get_normalization_factors(landmarks):
    nose = np.array([landmarks[0].x, landmarks[0].y])
    left_eye = np.array([landmarks[2].x, landmarks[2].y])
    right_eye = np.array([landmarks[5].x, landmarks[5].y])
    head_width = np.linalg.norm(left_eye - right_eye)
    if head_width == 0: head_width = 0.01
    return nose, head_width

def normalize_coordinates(landmark, nose, head_width):
    norm_x = (landmark.x - nose[0]) / (head_width * 7)
    norm_y = (landmark.y - nose[1]) / (head_width * 8)
    return [norm_x, norm_y]

def process_landmarks(results):
    if not results.pose_landmarks: return np.zeros(150)

    nose, head_width = get_normalization_factors(results.pose_landmarks.landmark)
    
    pose_flat = []
    for lm in results.pose_landmarks.landmark:
        n = normalize_coordinates(lm, nose, head_width)
        pose_flat.extend(n)
    
    lh_flat = []
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            n = normalize_coordinates(lm, nose, head_width)
            lh_flat.extend(n)
    else:
        lh_flat = [0] * 42

    rh_flat = []
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            n = normalize_coordinates(lm, nose, head_width)
            rh_flat.extend(n)
    else:
        rh_flat = [0] * 42
    
    return np.concatenate([pose_flat, lh_flat, rh_flat])

def extract_keypoints_for_visualization(results, bbox):
    keypoints = {
        'pose': None,
        'left_hand': None,
        'right_hand': None
    }
    
    if results.pose_landmarks:
        keypoints['pose'] = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z, 'visibility': lm.visibility}
            for lm in results.pose_landmarks.landmark
        ]
    
    if results.left_hand_landmarks:
        keypoints['left_hand'] = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z}
            for lm in results.left_hand_landmarks.landmark
        ]
    
    if results.right_hand_landmarks:
        keypoints['right_hand'] = [
            {'x': lm.x, 'y': lm.y, 'z': lm.z}
            for lm in results.right_hand_landmarks.landmark
        ]
    
    return keypoints

def process_frame(frame_data, client_id):
    frame = None
    img_data = None
    nparr = None
    image_rgb = None
    results_yolo = None
    df_yolo = None
    process_frame_crop = None
    img_crop_rgb = None
    input_tensor = None
    prediction = None
    probs = None
    
    try:
        try:
            gc.collect()
            
            if ',' in frame_data:
                frame_data = frame_data.split(',')[1]
            
            img_data = base64.b64decode(frame_data)
            
            try:
                nparr = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except MemoryError as me:
                print(f"Memory error during image decode, attempting cleanup: {me}")
                del img_data
                gc.collect()
                img_data = base64.b64decode(frame_data.split(',')[-1] if ',' in frame_data else frame_data)
                nparr = np.frombuffer(img_data, dtype=np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            del img_data
            del nparr
            
            if frame is None:
                return {
                    'detected_word': '...',
                    'confidence': 0.0,
                    'bounding_box': [0, 0, 0, 0],
                    'keypoints': {'pose': None, 'left_hand': None, 'right_hand': None},
                    'sequence_length': len(client_sequences.get(client_id, deque(maxlen=30)))
                }
        except MemoryError as me:
            print(f"Memory allocation error: {me}")
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return {
                'detected_word': '...',
                'confidence': 0.0,
                'bounding_box': [0, 0, 0, 0],
                'keypoints': {'pose': None, 'left_hand': None, 'right_hand': None},
                'sequence_length': len(client_sequences.get(client_id, deque(maxlen=30)))
            }
        except Exception as e:
            print(f"Error decoding image: {e}")
            return {
                'detected_word': '...',
                'confidence': 0.0,
                'bounding_box': [0, 0, 0, 0],
                'keypoints': {'pose': None, 'left_hand': None, 'right_hand': None},
                'sequence_length': len(client_sequences.get(client_id, deque(maxlen=30)))
            }
        
        if client_id not in client_sequences:
            client_sequences[client_id] = deque(maxlen=30)
        
        sequence = client_sequences[client_id]
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_yolo = model_yolo(image_rgb)
        h, w, _ = frame.shape
        x1, y1, x2, y2 = 0, 0, w, h
        
        df_yolo = results_yolo.pandas().xyxy[0]
        if not df_yolo.empty:
            best = df_yolo.iloc[0]
            best_confidence = best['confidence']
            best_xmin = best.xmin
            best_ymin = best.ymin
            best_xmax = best.xmax
            best_ymax = best.ymax
            del df_yolo
            
            if best_confidence > 0.4:
                pad = 20
                x1 = max(0, int(best_xmin) - pad)
                y1 = max(0, int(best_ymin) - pad)
                x2 = min(w, int(best_xmax) + pad)
                y2 = min(h, int(best_ymax) + pad)
        else:
            del df_yolo
        
        del results_yolo
        del image_rgb
        
        process_frame_crop = frame[y1:y2, x1:x2]
        if process_frame_crop.size == 0:
            process_frame_crop = frame
        
        img_crop_rgb = cv2.cvtColor(process_frame_crop, cv2.COLOR_BGR2RGB)
        results = holistic.process(img_crop_rgb)
        
        del process_frame_crop
        del img_crop_rgb
        
        keypoints = extract_keypoints_for_visualization(results, [x1, y1, x2, y2])
        
        detected_word = "..."
        confidence = 0.0
        
        if results.pose_landmarks:
            keypoints_features = process_landmarks(results)
            sequence.append(keypoints_features)
            
            if len(sequence) == 30:
                try:
                    input_seq = np.array(sequence, dtype=np.float32)
                except MemoryError:
                    print("Memory error creating input sequence, skipping prediction")
                    input_seq = None
                
                if input_seq is not None:
                    try:
                        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            prediction = model(input_tensor)
                            probs = torch.softmax(prediction, dim=1)
                            conf, predicted_idx = torch.max(probs, 1)
                            
                            confidence = conf.item()
                            if conf.item() > CONFIDENCE_THRESHOLD:
                                detected_word = CLASSES[predicted_idx.item()]
                        
                        del input_tensor
                        del prediction
                        del probs
                    except RuntimeError as re:
                        if "out of memory" in str(re).lower():
                            print(f"CUDA/CPU out of memory during inference: {re}")
                            if device.type == 'cuda':
                                torch.cuda.empty_cache()
                        else:
                            raise
                    finally:
                        del input_seq
                
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        result = {
            'detected_word': detected_word,
            'confidence': float(confidence),
            'bounding_box': [int(x1), int(y1), int(x2), int(y2)],
            'keypoints': keypoints,
            'sequence_length': len(sequence),
            'frame_width': int(w),
            'frame_height': int(h)
        }
        
        del frame
        
        gc.collect()
        
        return result
        
    except Exception as e:
        try:
            del frame
        except:
            pass
        try:
            del img_data
        except:
            pass
        try:
            del nparr
        except:
            pass
        try:
            del image_rgb
        except:
            pass
        try:
            del results_yolo
        except:
            pass
        try:
            del df_yolo
        except:
            pass
        try:
            del process_frame_crop
        except:
            pass
        try:
            del img_crop_rgb
        except:
            pass
        try:
            del input_tensor
        except:
            pass
        try:
            del prediction
        except:
            pass
        try:
            del probs
        except:
            pass
        gc.collect()
        raise e

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/health', methods=['GET'])
def health_check():
    return {
        'status': 'healthy',
        'service': 'sign-language-detector-backend',
        'models_loaded': model is not None and model_yolo is not None
    }, 200

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")
    client_sequences[request.sid] = deque(maxlen=30)
    processing_locks[request.sid] = False
    frame_queues[request.sid] = Queue(maxsize=1)

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")
    if request.sid in client_sequences:
        del client_sequences[request.sid]
    if request.sid in processing_locks:
        del processing_locks[request.sid]
    if request.sid in frame_queues:
        try:
            while True:
                frame_queues[request.sid].get_nowait()
        except Empty:
            pass
        del frame_queues[request.sid]

@socketio.on('frame')
def handle_frame(data):
    try:
        frame_data = data.get('data', '')
        if not frame_data:
            return
        
        client_id = request.sid
        
        if client_id not in frame_queues:
            frame_queues[client_id] = Queue(maxsize=1)
            processing_locks[client_id] = False
        
        try:
            frame_queues[client_id].put_nowait(frame_data)
        except:
            try:
                frame_queues[client_id].get_nowait()
                frame_queues[client_id].put_nowait(frame_data)
            except:
                pass
        
        if processing_locks[client_id]:
            return
        
        try:
            latest_frame = frame_queues[client_id].get_nowait()
        except Empty:
            return
        
        processing_locks[client_id] = True
        
        try:
            result = process_frame(latest_frame, client_id)
            emit('result', result)
        finally:
            processing_locks[client_id] = False
            
    except Exception as e:
        print(f"Error processing frame: {e}")
        emit('error', {'message': str(e)})
        if request.sid in processing_locks:
            processing_locks[request.sid] = False
        gc.collect()

@socketio.on('chat_message')
def handle_chat_message(data):
    try:
        message_text = data.get('text', '').strip()
        if not message_text:
            return
        
        import time
        message_data = {
            'text': message_text,
            'sender': 'user',
            'timestamp': int(time.time() * 1000)
        }
        
        socketio.emit('chat_message', message_data)
        print(f"Chat message broadcast: {message_text[:50]}...")
        
    except Exception as e:
        print(f"Error handling chat message: {e}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    import os
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.getenv('PORT', '5000'))
    print(f"Starting WebSocket server on http://0.0.0.0:{port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=debug_mode)

