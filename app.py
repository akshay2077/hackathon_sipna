from flask import Flask, request, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import uuid
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Flask app setup
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load models
model = tf.keras.models.load_model('model.keras')
text_model = joblib.load('device_recommender.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocessing
stop_words = set(stopwords.words('english'))
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

# Class names
class_names = ['Battery', 'Keynoard', 'Microwave', 'Mobile', 'Mouse', 'PCB',
               'Player', 'Printer', 'Television', 'Washing Machine']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    uploaded_image = None
    show_extra_form = False
    recommendation_done = False

    if request.method == 'POST':
        file = request.files.get('file')

        if file and file.filename != '':
            filename = f"{uuid.uuid4().hex}.jpg"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            image = Image.open(file.stream).convert("RGB")
            image = image.resize((224, 224))
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            preds = model.predict(image_array)
            confidence = np.max(preds)
            pred_class = class_names[np.argmax(preds)]

            if pred_class in class_names and confidence > 0.7:
                prediction = f"Predicted Device: {pred_class} ({confidence*100:.2f}%)"
                show_extra_form = True
            else:
                prediction = "Not an electronic device."
                show_extra_form = False

            uploaded_image = filename

    return render_template(
        'index.html',
        prediction=prediction,
        show_extra_form=show_extra_form,
        uploaded_image=uploaded_image,
        recommendation_done=recommendation_done
    )


@app.route('/extra-info', methods=['POST'])
def extra_info():
    info = request.form.get('additional_info')

    if info:
        processed = preprocess(info)
        tfidf_input = vectorizer.transform([processed])
        recommendation = text_model.predict(tfidf_input)[0]

        result_text = f"🔧 Based on your input, EcoBot recommends you to go for : <strong>{recommendation.upper()}</strong> the device."
        return render_template(
            'index.html',
            prediction=result_text,
            show_extra_form=False,
            uploaded_image=None,
            recommendation_done=True
        )

    return render_template(
        'index.html',
        prediction="No additional info provided.",
        show_extra_form=False,
        uploaded_image=None,
        recommendation_done=True
    )


if __name__ == '__main__':
    app.run(debug=True)
