from flask import Flask, render_template, request, send_file, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
import firebase_admin
from firebase_admin import credentials, db
from datetime import datetime
from collections import Counter
from time import time
from flask import Response
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
LOG_FILE = 'prediction_log.csv'

# Initialize Firebase app
cred = credentials.Certificate('plantdetectionsystem-c6c3a-firebase-adminsdk-fbsvc-2d4ea6333a.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://plantdetectionsystem-c6c3a-default-rtdb.firebaseio.com/'
})

# Crop to model and class mapping
CROP_MODELS = {
    'apple': ('models/apple_model.h5', ['Apple Scab', 'Black Rot', 'Rust', 'Healthy']),
    'tomato': ('models/tomato_model.h5', ['Bacterial Spot', 'Early Blight', 'Late Blight', 'Leaf Mold', 'Septoria Leaf Spot', 'Spider Mites', 'Target Spot', 'Mosaic Virus', 'Yellow Curl Virus', 'Healthy'])
}

# Global variables for live prediction
live_prediction_active = False
current_crop = None
current_model = None
current_classes = None

def init_model(crop):
    global current_model, current_classes, current_crop
    model_path, classes = CROP_MODELS.get(crop)
    current_model = tf.keras.models.load_model(model_path)
    current_classes = classes
    current_crop = crop
    return current_model, current_classes

def predict_disease(img_path, model, classes, threshold=70):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)

    if len(prediction[0]) != len(classes):
        raise ValueError(f"Model output size ({len(prediction[0])}) doesn't match number of class labels ({len(classes)})")

    idx = np.argmax(prediction[0])
    confidence = round(prediction[0][idx] * 100, 2)

    if confidence < threshold:
        return "Unknown", confidence
    else:
        return classes[idx], confidence

def log_prediction(crop, label, confidence, filename, duration):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'timestamp': timestamp,
        'crop': crop,
        'label': label,
        'confidence': confidence,
        'image': filename,
        'duration': duration
    }

    df = pd.DataFrame([data])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    ref = db.reference('predictions')
    ref.push(data)

def get_recent_predictions(n=5):
    if not os.path.exists(LOG_FILE):
        return []
    df = pd.read_csv(LOG_FILE).tail(n)
    return df.to_dict(orient='records')

RECOMMENDATIONS = {
    'apple': {
        "Apple Scab": "Use a protective fungicide and ensure proper pruning.",
        "Black Rot": "Prune infected branches and apply fungicide.",
        "Rust": "Consider Mancozeb spray and isolate affected plants.",
        "Healthy": "No action needed. Plant appears healthy."
    },
    'tomato': {
        "Early Blight": "Apply appropriate fungicide (e.g., Chlorothalonil) and remove affected leaves.",
        "Late Blight": "Remove and destroy affected plants, apply protective sprays.",
        "Healthy": "No action needed. Plant appears healthy."
        
    }
}

def gen_frames():
    global live_prediction_active, current_model, current_classes, current_crop
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Camera not detected.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame, (224, 224))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Green mask
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Contours for bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        leaf_detected = False

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1500:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                leaf_detected = True

        # Predict if leaf detected and prediction is active
        label_text = "Select crop and click 'Predict Disease' to start" if not live_prediction_active else "Detecting..."
        
        if leaf_detected and live_prediction_active and current_model is not None:
            img_array = image.img_to_array(resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            prediction = current_model.predict(img_array)[0]
            idx = np.argmax(prediction)
            confidence = round(prediction[idx] * 100, 2)
            label_text = f"{current_classes[idx]} ({confidence}%)"

            if confidence > 75:
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_live.jpg"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                cv2.imwrite(filepath, frame)
                log_prediction(current_crop, current_classes[idx], confidence, filename, duration=0)
                label_text += " ✅ Saved"

        cv2.putText(resized, label_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', resized)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/', methods=['GET', 'POST'])
def index():
    global live_prediction_active, current_crop
    
    prediction = None
    image_path = None
    crop = None
    history = get_recent_predictions()

    if request.method == 'POST':
        input_method = request.form.get('input_method')
        crop = request.form.get('crop')

        model_path, classes = CROP_MODELS.get(crop, (None, None))
        if not model_path:
            return render_template("index.html", error="Invalid crop selected.")

        if input_method == 'upload':
            live_prediction_active = False
            file = request.files.get('image')
            if file:
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
                filepath = os.path.join(UPLOAD_FOLDER, filename)
                file.save(filepath)
                
                model = tf.keras.models.load_model(model_path)
                start_time = time()
                label, confidence = predict_disease(filepath, model, classes)
                duration = round(time() - start_time, 2)
                prediction = f"{label} ({confidence}%)"
                image_path = filename
                log_prediction(crop, label, confidence, filename, duration)
                history = get_recent_predictions()

        elif input_method == 'webcam':
            live_prediction_active = True
            init_model(crop)
            # No immediate prediction, just start the live feed
            return render_template("index.html", prediction=None, image_path=None, 
                                history=history, crop=crop, recommendations=RECOMMENDATIONS,
                                live_feed_active=True)
        else:
            return render_template("index.html", error="Invalid input method selected.")

    return render_template("index.html", prediction=prediction, image_path=image_path, 
                         history=history, crop=crop, recommendations=RECOMMENDATIONS,
                         live_feed_active=live_prediction_active)

@app.route('/graphs')
def graphs():
    if not os.path.exists(LOG_FILE):
        return render_template("graphs.html", labels=[], counts=[], dates=[], timeline=[], top_disease="N/A", top_disease_count=0, avg_time=0)

    df = pd.read_csv(LOG_FILE)

    labels = df['label'].value_counts().index.tolist()
    counts = df['label'].value_counts().values.tolist()

    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    date_counts = df.groupby('date').size()
    dates = list(map(str, date_counts.index))
    timeline = date_counts.tolist()

    top_disease_data = df['label'].value_counts()
    top_disease = top_disease_data.index[0]
    top_disease_count = int(top_disease_data.iloc[0])

    if 'duration' in df.columns:
        times = df['duration'].dropna().astype(float).tolist()
        avg_time = round(sum(times) / len(times), 2) if times else 0
    else:
        avg_time = "N/A"

    return render_template("graphs.html",
                         labels=labels,
                         counts=counts,
                         dates=dates,
                         timeline=timeline,
                         top_disease=top_disease,
                         top_disease_count=top_disease_count,
                         avg_time=avg_time)

@app.route('/history')
def history():
    if not os.path.exists(LOG_FILE):
        return render_template("history.html", history=[])
    df = pd.read_csv(LOG_FILE)
    return render_template("history.html", history=df.to_dict(orient='records'))

@app.route('/download_log')
def download_log():
    return send_file(LOG_FILE, as_attachment=True)

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)