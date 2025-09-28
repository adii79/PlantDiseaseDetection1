import warnings
warnings.filterwarnings("ignore")  # hide urllib3 + Tk deprecation warnings

import os
os.environ["TK_SILENCE_DEPRECATION"] = "1"  # silence macOS Tk warning

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import Label, Button
from PIL import Image, ImageTk
import numpy as np
import json
import tensorflow as tf

# ===============================
# Load Model and Labels
# ===============================
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

label = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Background_without_leaves',
 'Blueberry___healthy',
 'Cherry___Powdery_mildew',
 'Cherry___healthy',
 'Corn___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn___Common_rust',
 'Corn___Northern_Leaf_Blight',
 'Corn___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

with open("plant_disease.json",'r') as file:
    plant_disease = json.load(file)

# ===============================
# Feature Extraction
# ===============================
def extract_features(image_path):
    image = tf.keras.utils.load_img(image_path, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.array([feature])
    return feature

def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label

# ===============================
# Tkinter GUI
# ===============================
class PlantDiseaseApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Plant Disease Detection")
        self.root.geometry("600x550")
        self.root.config(bg="#f2f2f2")

        self.file_path = None  # store chosen image

        # Heading
        self.label = Label(root, text="🌱 Plant Disease Recognition 🌱", font=("Arial", 18, "bold"), bg="#f2f2f2")
        self.label.pack(pady=10)

        # Image Preview
        self.image_label = Label(root, bg="#ffffff", width=300, height=300)
        self.image_label.pack(pady=10)

        # Prediction Label
        self.result_label = Label(root, text="", font=("Arial", 14), bg="#f2f2f2", fg="green")
        self.result_label.pack(pady=10)

        # Buttons
        self.choose_btn = Button(root, text="Choose Image", command=self.choose_image,
                                 font=("Arial", 12), bg="#4CAF50", fg="white")
        self.choose_btn.pack(pady=5)

        self.predict_btn = Button(root, text="Predict Disease", command=self.predict_image,
                                  font=("Arial", 12), bg="#2196F3", fg="white")
        self.predict_btn.pack(pady=5)

        self.exit_btn = Button(root, text="Exit", command=root.quit,
                               font=("Arial", 12), bg="red", fg="white")
        self.exit_btn.pack(pady=5)

    def choose_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_path:
            return

        self.file_path = file_path

        # Display chosen image
        img = Image.open(file_path)
        img = img.resize((300, 300))
        self.img_tk = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.img_tk)

        # Reset previous result
        self.result_label.config(text="")

    def predict_image(self):
        if not self.file_path:
            messagebox.showwarning("No Image", "Please choose an image first.")
            return

        try:
            prediction = model_predict(self.file_path)
            self.result_label.config(text=f"Prediction: {prediction}", fg="blue")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to predict: {e}")

# ===============================
# Run App
# ===============================
if __name__ == "__main__":
    root = tk.Tk()
    app = PlantDiseaseApp(root)
    root.mainloop()
