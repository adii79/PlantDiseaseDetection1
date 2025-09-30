import tensorflow as tf

# ------------------------
# Load the SavedModel/Keras model
# ------------------------
model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")

# ------------------------
# Convert to HDF5 (.h5)
# ------------------------
model.save("plant_disease_recog_model_pwp.h5")
print("✅ Saved model in HDF5 format as plant_disease_recog_model_pwp.h5")

# ------------------------
# Convert to TensorFlow Lite (.tflite)
# ------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional optimizations (smaller model, useful for mobile/IoT)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert and save
tflite_model = converter.convert()
with open("plant_disease_recog_model_pwp.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved model in TensorFlow Lite format as plant_disease_recog_model_pwp.tflite")
