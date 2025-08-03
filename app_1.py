from flask import Flask, render_template, request,  send_file, redirect, url_for
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


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
LOG_FILE = 'prediction_log.csv'

UPLOAD_FOLDER = 'static/uploads'


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



def predict_disease(img_path, model, classes, threshold=70):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    
    if len(prediction[0]) != len(classes):
        raise ValueError(f"Model output size ({len(prediction[0])}) doesn't match number of class labels ({len(classes)})")
    
    idx = np.argmax(prediction[0])
    confidence = round(prediction[0][idx] * 100, 2)
    
# ✅ Confidence threshold logic
    if confidence < threshold:
        return "Unknown", confidence
    else:
        return classes[idx], confidence

def log_prediction(crop, label, confidence, filename,duration):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    data = {
        'timestamp': timestamp,
        'crop': crop,
        'label': label,
        'confidence': confidence,
        'image': filename,
        'duration': duration
    }

    # Save to CSV
    df = pd.DataFrame([data])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    # Save to Firebase
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



@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None
    crop = None
    history = get_recent_predictions()

    if request.method == 'POST':
        crop = request.form['crop']
        file = request.files['image']
        if crop in CROP_MODELS and file:
            model_path, classes = CROP_MODELS[crop]
            model = tf.keras.models.load_model(model_path)

            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # ✅ Start timer
            start_time = time()
            label, confidence = predict_disease(filepath, model, classes)
            duration = round(time() - start_time, 2)  # ✅ End timer

            prediction = f"{label} ({confidence}%)"
            image_path = filename

            # ✅ Pass duration to log_prediction
            log_prediction(crop, label, confidence, filename, duration)
            history = get_recent_predictions()

    return render_template("index.html", prediction=prediction, image_path=image_path, history=history, crop=crop, recommendations=RECOMMENDATIONS)

@app.route('/graphs')
def graphs():
    if not os.path.exists(LOG_FILE):
        return render_template("graphs.html", labels=[], counts=[], dates=[], timeline=[], top_disease="N/A", top_disease_count=0, avg_time=0)

    df = pd.read_csv(LOG_FILE)

    # Disease count for bar chart
    labels = df['label'].value_counts().index.tolist()
    counts = df['label'].value_counts().values.tolist()

    # Daily prediction count for line chart
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    date_counts = df.groupby('date').size()
    dates = list(map(str, date_counts.index))
    timeline = date_counts.tolist()

    # ✅ Top detected disease
    top_disease_data = df['label'].value_counts()
    top_disease = top_disease_data.index[0]
    top_disease_count = int(top_disease_data.iloc[0])

    # ✅ Average detection time \
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


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

