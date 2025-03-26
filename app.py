from flask import Flask, request, render_template, jsonify  
import subprocess
import os
import pickle
import base64
import io
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO  # YOLOv8 for object detection
import face_recognition
from deepface import DeepFace  # For emotion detection

app = Flask(__name__)

# Load BLIP model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load YOLO model for object detection
yolo_model = YOLO("yolov8n.pt")

# Directory for known faces
KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)

def load_known_faces():
    """Load known faces from storage."""
    known_faces = {}
    for file_name in os.listdir(KNOWN_FACES_DIR):
        name = os.path.splitext(file_name)[0]
        file_path = os.path.join(KNOWN_FACES_DIR, file_name)
        with open(file_path, 'rb') as f:
            encoding = pickle.load(f)
            known_faces[name] = encoding
    return known_faces

def detect_objects(image):
    """Detect objects in the image using YOLOv8."""
    results = yolo_model(image)
    object_labels = set()
    for result in results:
        for item in result.boxes.data.tolist():
            label = int(item[5])  # The class index
            object_labels.add(result.names[label])  # Convert class index to name
    return list(object_labels)

def generate_final_sentence(caption, objects, face_results):
    """
    Generate a sentence like:
    "We are seeing Manikandan with a red shirt. Featuring objects such as chair.
     Manikandan seems neutral."
    """
    # Clean up the caption:
    caption = caption.strip().rstrip('.')
    # Remove leading "a " or "an " if present
    if caption.lower().startswith("a "):
        caption = caption[2:].strip()
    elif caption.lower().startswith("an "):
        caption = caption[3:].strip()

    sentence_parts = []

    # Sentence 1: Visual description
    sentence1 = f"We are seeing {caption}."
    sentence_parts.append(sentence1)

    # Sentence 2: Non-person objects (if any)
    non_person_objects = [obj for obj in objects if obj.lower() != "person"]
    if non_person_objects:
        sentence2 = f"Featuring objects such as {', '.join(non_person_objects)}."
        sentence_parts.append(sentence2)

    # Sentence 3: People and their emotions (from face recognition)
    if face_results:
        if len(face_results) == 1:
            # For single person, just mention their emotion
            face = face_results[0]
            sentence3 = f"{face['name']} seems {face['emotion']}."
        else:
            # For multiple people, include count and all emotions
            count = len(face_results)
            face_names = ", ".join(face['name'] for face in face_results)
            sentence3 = f"There are {count} persons including {face_names}"
            emotion_details = ", ".join([f"{face['name']} seems {face['emotion']}" for face in face_results])
            sentence3 += f", {emotion_details}."
        sentence_parts.append(sentence3)

    final_sentence = " ".join(sentence_parts)
    return final_sentence

@app.route('/')
def landing_page():
    return render_template('index.html')

@app.route('/image-captioning')
def image_captioning():
    return render_template('image_captioning.html')

@app.route('/add-face')
def add_face():
    return render_template('add_face.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload, perform captioning, object detection, and face recognition with emotion detection."""
    try:
        img_data = request.json['image']
        img_bytes = base64.b64decode(img_data.split(',')[1])
        image = Image.open(io.BytesIO(img_bytes))

        # Object Detection
        objects = detect_objects(image)

        # Face Detection
        image_np = face_recognition.load_image_file(io.BytesIO(img_bytes))
        face_locations = face_recognition.face_locations(image_np)
        face_encodings = face_recognition.face_encodings(image_np, face_locations)

        known_faces = load_known_faces()
        face_results = []

        # Identify faces & detect emotions
        for encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            match = face_recognition.compare_faces(list(known_faces.values()), encoding, tolerance=0.6)
            name = "Unknown"
            if any(match):
                name = list(known_faces.keys())[match.index(True)]

            # Crop face for emotion analysis
            face_crop = image_np[top:bottom, left:right]
            try:
                emotion_analysis = DeepFace.analyze(face_crop, actions=['emotion'], enforce_detection=False)
                if isinstance(emotion_analysis, list):
                    emotion = emotion_analysis[0]['dominant_emotion']
                else:
                    emotion = emotion_analysis['dominant_emotion']
            except Exception:
                emotion = "Unknown"

            face_results.append({"name": name, "emotion": emotion})

        # Generate Image Caption
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        # Optionally replace generic words (e.g., "man"/"woman") in the caption with detected face names
        for face in face_results:
            name = face["name"]
            caption = caption.replace("woman", name)  # Add "Wo" prefix for women
            caption = caption.replace("man", name)  # Keep original name for men

        # Create a combined sentence using our new function
        final_sentence = generate_final_sentence(caption, objects, face_results)

        return jsonify({
            "sentence": final_sentence
        })

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/query', methods=['POST'])
def query_model():
    """Process user queries using the Ollama CLI with the llama3.2:latest model."""
    data = request.json
    user_query = data.get('query', '')

    if not user_query:
        return jsonify({"error": "Query is required"})

    try:
        result = subprocess.run(
            ['ollama', 'run', 'llama3.2:latest', user_query],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode != 0:
            return jsonify({"error": f"Failed to interact with Ollama: {result.stderr}"})

        return jsonify({"response": result.stdout.strip()})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/save-faces', methods=['POST'])
def save_faces():
    """Save multiple faces to the known faces directory."""
    try:
        if 'images' not in request.files or 'name' not in request.form:
            return "Missing images or name", 400

        name = request.form['name']
        images = request.files.getlist('images')
        
        if not images:
            return "No images provided", 400

        successful_encodings = []

        # Process each image
        for image in images:
            image_data = image.read()
            image_np = face_recognition.load_image_file(io.BytesIO(image_data))
            face_encodings = face_recognition.face_encodings(image_np)

            if face_encodings:
                successful_encodings.append(face_encodings[0])

        if not successful_encodings:
            return "No faces detected in any of the images", 400

        # Calculate average encoding
        average_encoding = sum(successful_encodings) / len(successful_encodings)

        # Save the average face encoding
        file_path = os.path.join(KNOWN_FACES_DIR, f"{name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(average_encoding, f)

        return f"Successfully saved {len(successful_encodings)} faces for {name}", 200

    except Exception as e:
        return f"Error saving faces: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)
