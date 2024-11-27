import os 
import zipfile
from flask import Flask, request, render_template, send_from_directory
import numpy as np
import cv2
from keras.preprocessing import image
from keras.models import load_model

app = Flask(__name__)

# Define paths
MODEL_ZIP_PATH = 'model/best_hybrid_model.keras.zip'
MODEL_PATH = 'model/best_hybrid_model.keras'
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
EXTRACTED_FRAMES_FOLDER = 'extracted_frames'

if MODEL_ZIP_PATH.endswith('.zip'):
    with zipfile.ZipFile(MODEL_ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall('model')

# Load the model
model = load_model(MODEL_PATH)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(EXTRACTED_FRAMES_FOLDER, exist_ok=True)

# Function to prepare an image for prediction
def prepare_image(img):
    img = cv2.resize(img, (224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Function to extract frames from a video
def extract_frames(video_path, num_frames=5):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    for index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    
    if not frames:
        raise ValueError("Could not retrieve frames from video.")
    
    return frames

# Function to draw a rectangle around the face
def draw_face_rectangle(img):
    x, y, w, h = 50, 50, 100, 100 
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result='Error: No file part')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', result='Error: No selected file')

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    
    processed_image_path = None
    predictions = []
    extracted_frames_paths = []

    if is_video:
        try:
            frames = extract_frames(file_path, num_frames=5)  # Extract 5 frames
            for idx, frame in enumerate(frames):
                img_array = prepare_image(frame)
                processed_frame = draw_face_rectangle(frame)  # Draw rectangle on frame
                processed_frame_path = os.path.join(EXTRACTED_FRAMES_FOLDER, f'frame_{idx}.jpg')
                cv2.imwrite(processed_frame_path, processed_frame)
                extracted_frames_paths.append(f'frame_{idx}.jpg')  # Store just the filename

                # Predict on the frame
                prediction = model.predict(img_array)[0][0]
                predictions.append(prediction)

        except Exception as e:
            return render_template('index.html', result=f"Error processing video: {e}")
    else:
        img = cv2.imread(file_path)
        img_array = prepare_image(img)
        processed_frame = draw_face_rectangle(img)  # Draw rectangle on image
        
        # Save the processed image
        processed_image_path = os.path.join(PROCESSED_FOLDER, file.filename)
        cv2.imwrite(processed_image_path, processed_frame)

        # Predict on the image
        prediction = model.predict(img_array)[0][0]
        predictions.append(prediction)
    if is_video:
        average_prediction = np.mean(predictions)
    else:
        average_prediction = predictions[0]

    threshold = 0.5  
    if is_video:
        if average_prediction < threshold:
            label = 'Real'
            confidence = 1 - average_prediction 
        else:
            label = 'Fake'
            confidence = average_prediction
    else:
        if average_prediction < threshold:
            label = 'Fake'
            confidence = 1 - average_prediction
        else:
            label = 'Real'
            confidence = average_prediction

    confidence_percentage = round(confidence * 100, 2)
    raw_prediction_percentage = round(average_prediction * 100, 2)
    explanation = (
        f"The model predicts this as '{label}' with a confidence of {confidence_percentage}%. "
        f"(Raw prediction: {raw_prediction_percentage}%)"
    )

    return render_template('index.html', result=label, confidence=f"{confidence_percentage}%", 
                           explanation=explanation, filename=file.filename, processed_image=processed_image_path,
                           extracted_frames=extracted_frames_paths)

@app.route('/processed/<filename>')
def processed_image(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route('/extracted_frames/<filename>')
def extracted_frame(filename):
    return send_from_directory(EXTRACTED_FRAMES_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
