import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# ---------------------
# FLASK APP
# ---------------------
app = Flask(__name__)

# ---------------------
# SET UPLOAD FOLDER (IMPORTANT: top of file)
# ---------------------
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------------------
# LOAD MODEL
# ---------------------
MODEL_PATH = "model/mask_model_final.keras"

print("\nLoading model from:", os.path.abspath(MODEL_PATH))
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully!\n")


# ---------------------
# PREDICTION FUNCTION
# ---------------------
def predict_mask(img_path):
    img = image.load_img(img_path, target_size=(224,224))
    img = image.img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        return "Mask Detected", pred
    else:
        return "No Mask Detected", 1 - pred


# ---------------------
# ROUTES
# ---------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"})

    filename = file.filename.replace(" ", "_")
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    label, conf = predict_mask(save_path)
    conf_percent = f"{conf*100:.2f}%"

    url_path = "/" + save_path.replace("\\", "/")

    return jsonify({
        "status": "success",
        "label": label,
        "confidence": conf_percent,
        "image_path": url_path
    })


# ---------------------
# RUN SERVER
# ---------------------
if __name__ == "__main__":
    app.run(debug=True)
