# from flask import Flask, render_template,request,redirect,send_from_directory,url_for
# import numpy as np
# import json
# import uuid
# import tensorflow as tf

# app = Flask(__name__)
# model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
# label = ['Apple___Apple_scab',
#  'Apple___Black_rot',
#  'Apple___Cedar_apple_rust',
#  'Apple___healthy',
#  'Background_without_leaves',
#  'Blueberry___healthy',
#  'Cherry___Powdery_mildew',
#  'Cherry___healthy',
#  'Corn___Cercospora_leaf_spot Gray_leaf_spot',
#  'Corn___Common_rust',
#  'Corn___Northern_Leaf_Blight',
#  'Corn___healthy',
#  'Grape___Black_rot',
#  'Grape___Esca_(Black_Measles)',
#  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#  'Grape___healthy',
#  'Orange___Haunglongbing_(Citrus_greening)',
#  'Peach___Bacterial_spot',
#  'Peach___healthy',
#  'Pepper,_bell___Bacterial_spot',
#  'Pepper,_bell___healthy',
#  'Potato___Early_blight',
#  'Potato___Late_blight',
#  'Potato___healthy',
#  'Raspberry___healthy',
#  'Soybean___healthy',
#  'Squash___Powdery_mildew',
#  'Strawberry___Leaf_scorch',
#  'Strawberry___healthy',
#  'Tomato___Bacterial_spot',
#  'Tomato___Early_blight',
#  'Tomato___Late_blight',
#  'Tomato___Leaf_Mold',
#  'Tomato___Septoria_leaf_spot',
#  'Tomato___Spider_mites Two-spotted_spider_mite',
#  'Tomato___Target_Spot',
#  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#  'Tomato___Tomato_mosaic_virus',
#  'Tomato___healthy']

# with open("plant_disease.json",'r') as file:
#     plant_disease = json.load(file)

# # print(plant_disease[4])

# @app.route('/uploadimages/<path:filename>')
# def uploaded_images(filename):
#     return send_from_directory('./uploadimages', filename)

# @app.route('/',methods = ['GET'])
# def home():
#     return render_template('home.html')

# def extract_features(image):
#     image = tf.keras.utils.load_img(image,target_size=(160,160))
#     feature = tf.keras.utils.img_to_array(image)
#     feature = np.array([feature])
#     return feature

# def model_predict(image):
#     img = extract_features(image)
#     prediction = model.predict(img)
#     # print(prediction)
#     prediction_label = plant_disease[prediction.argmax()]
#     return prediction_label

# @app.route('/upload/',methods = ['POST','GET'])
# def uploadimage():
#     if request.method == "POST":
#         image = request.files['img']
#         temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
#         image.save(f'{temp_name}_{image.filename}')
#         print(f'{temp_name}_{image.filename}')
#         prediction = model_predict(f'./{temp_name}_{image.filename}')
#         return render_template('home.html',result=True,imagepath = f'/{temp_name}_{image.filename}', prediction = prediction )
    
#     else:
#         return redirect('/')
        
    
# if __name__ == "__main__":
#     app.run(debug=True)





# from flask import Flask, render_template, request, redirect, send_from_directory, url_for
# import numpy as np
# import json
# import uuid
# import tensorflow as tf
# from rembg import remove
# from PIL import Image
# import os

# app = Flask(__name__)

# # Load model
# model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# # Labels
# label = ['Apple___Apple_scab',
#  'Apple___Black_rot',
#  'Apple___Cedar_apple_rust',
#  'Apple___healthy',
#  'Background_without_leaves',
#  'Blueberry___healthy',
#  'Cherry___Powdery_mildew',
#  'Cherry___healthy',
#  'Corn___Cercospora_leaf_spot Gray_leaf_spot',
#  'Corn___Common_rust',
#  'Corn___Northern_Leaf_Blight',
#  'Corn___healthy',
#  'Grape___Black_rot',
#  'Grape___Esca_(Black_Measles)',
#  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#  'Grape___healthy',
#  'Orange___Haunglongbing_(Citrus_greening)',
#  'Peach___Bacterial_spot',
#  'Peach___healthy',
#  'Pepper,_bell___Bacterial_spot',
#  'Pepper,_bell___healthy',
#  'Potato___Early_blight',
#  'Potato___Late_blight',
#  'Potato___healthy',
#  'Raspberry___healthy',
#  'Soybean___healthy',
#  'Squash___Powdery_mildew',
#  'Strawberry___Leaf_scorch',
#  'Strawberry___healthy',
#  'Tomato___Bacterial_spot',
#  'Tomato___Early_blight',
#  'Tomato___Late_blight',
#  'Tomato___Leaf_Mold',
#  'Tomato___Septoria_leaf_spot',
#  'Tomato___Spider_mites Two-spotted_spider_mite',
#  'Tomato___Target_Spot',
#  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#  'Tomato___Tomato_mosaic_virus',
#  'Tomato___healthy']

# # Load mapping JSON
# with open("plant_disease.json", 'r') as file:
#     plant_disease = json.load(file)


# @app.route('/uploadimages/<path:filename>')
# def uploaded_images(filename):
#     return send_from_directory('./uploadimages', filename)


# @app.route('/', methods=['GET'])
# def home():
#     return render_template('home.html')


# def remove_background(input_path, output_path):
#     """Remove background and make it white."""
#     input_image = Image.open(input_path).convert("RGBA")
#     result = remove(input_image)  # transparent background

#     # Make background white
#     white_bg = Image.new("RGBA", result.size, (255, 255, 255, 255))
#     final_img = Image.alpha_composite(white_bg, result)
#     final_img = final_img.convert("RGB")  # drop alpha for model

#     final_img.save(output_path)
#     return output_path


# def extract_features(image):
#     image = tf.keras.utils.load_img(image, target_size=(160, 160))
#     feature = tf.keras.utils.img_to_array(image)
#     feature = np.array([feature])
#     return feature


# def model_predict(image):
#     img = extract_features(image)
#     prediction = model.predict(img)
#     prediction_label = plant_disease[prediction.argmax()]
#     return prediction_label


# @app.route('/upload/', methods=['POST', 'GET'])
# def uploadimage():
#     if request.method == "POST":
#         image = request.files['img']
#         os.makedirs("uploadimages", exist_ok=True)

#         # Save original upload
#         temp_name = f"uploadimages/temp_{uuid.uuid4().hex}_{image.filename}"
#         image.save(temp_name)

#         # Remove background → save processed
#         processed_path = temp_name.replace("temp_", "processed_")
#         processed_path = remove_background(temp_name, processed_path)

#         # Run prediction on processed image
#         prediction = model_predict(processed_path)

#         return render_template(
#             'home.html',
#             result=True,
#             imagepath=f'/{processed_path}',
#             prediction=prediction
#         )
#     else:
#         return redirect('/')


# if __name__ == "__main__":
#     app.run(debug=True)





from flask import Flask, render_template, request, redirect, send_from_directory
import numpy as np
import json
import uuid
import tensorflow as tf
from rembg import remove
from PIL import Image
import os

app = Flask(__name__)

# ✅ Load the original Keras model
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# Labels
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

# Load mapping JSON
with open("plant_disease.json", 'r') as file:
    plant_disease = json.load(file)


@app.route('/uploadimages/<path:filename>')
def uploaded_images(filename):
    return send_from_directory('./uploadimages', filename)


@app.route('/', methods=['GET'])
def home():
    return render_template('home.html')


def remove_background(input_path, output_path):
    """Remove background and make it white."""
    input_image = Image.open(input_path).convert("RGBA")
    result = remove(input_image)  # transparent background

    # Make background white
    white_bg = Image.new("RGBA", result.size, (255, 255, 255, 255))
    final_img = Image.alpha_composite(white_bg, result)
    final_img = final_img.convert("RGB")  # drop alpha for model

    final_img.save(output_path)
    return output_path


def extract_features(image):
    image = tf.keras.utils.load_img(image, target_size=(160, 160))
    feature = tf.keras.utils.img_to_array(image)
    feature = np.expand_dims(feature, axis=0)  # add batch dimension
    return feature


def model_predict(image):
    img = extract_features(image)
    prediction = model.predict(img)
    prediction_label = plant_disease[prediction.argmax()]
    return prediction_label


@app.route('/upload/', methods=['POST', 'GET'])
def uploadimage():
    if request.method == "POST":
        image = request.files['img']
        os.makedirs("uploadimages", exist_ok=True)

        # Save original upload
        temp_name = f"uploadimages/temp_{uuid.uuid4().hex}_{image.filename}"
        image.save(temp_name)

        # Remove background → save processed
        processed_path = temp_name.replace("temp_", "processed_")
        processed_path = remove_background(temp_name, processed_path)

        # Run prediction on processed image
        prediction = model_predict(processed_path)

        return render_template(
            'home.html',
            result=True,
            imagepath=f'/{processed_path}',
            prediction=prediction
        )
    else:
        return redirect('/')


if __name__ == "__main__":
    app.run(debug=True)
