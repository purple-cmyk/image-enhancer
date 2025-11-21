import os
from flask import Flask, request, render_template_string
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

# --------------------------
# Load BLIP Model
# --------------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# --------------------------
# Flask App
# --------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --------------------------
# HTML Template (Inline)
# --------------------------
HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Image Description Generator</title>
    <style>
        body {
            background: #1c1f26;
            font-family: Arial, sans-serif;
            color: white;
            text-align: center;
            padding-top: 60px;
        }
        .container {
            width: 50%%;
            margin: auto;
            padding: 40px;
            background: #2b2f38;
            border-radius: 10px;
        }
        input[type=file] {
            padding: 10px;
            background: #fff;
            color: black;
            border-radius: 5px;
        }
        button {
            margin-top: 20px;
            padding: 12px 25px;
            font-size: 16px;
            border-radius: 5px;
            border: none;
            background: #4CAF50;
            cursor: pointer;
            color: white;
        }
        img {
            margin-top: 20px;
            width: 300px;
            border-radius: 8px;
        }
        .output {
            margin-top: 25px;
            padding: 20px;
            background: #3a3f4b;
            border-radius: 8px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h2>Image Description Generator (BLIP)</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="image" required>
            <br>
            <button type="submit">Generate Description</button>
        </form>

        {% if image_url %}
            <img src="{{ image_url }}">
        {% endif %}

        {% if caption %}
            <div class="output">
                <h3>Description:</h3>
                <p>{{ caption }}</p>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

# --------------------------
# BLIP Caption Function
# --------------------------
def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# --------------------------
# Routes
# --------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_url = None

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Generate caption
            caption = generate_caption(filepath)

            image_url = "/" + filepath

    return render_template_string(HTML_PAGE, caption=caption, image_url=image_url)

# --------------------------
# Serve Uploaded Files
# --------------------------
@app.route('/uploads/<path:filename>')
def uploaded(filename):
    return app.send_static_file(os.path.join("uploads", filename))

# --------------------------
# Run App
# --------------------------
if __name__ == "__main__":
    app.run(debug=True)
