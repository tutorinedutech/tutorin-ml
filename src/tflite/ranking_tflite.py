import tensorflow as tf
import numpy as np
model_path = './save_models/ranking_models/1'

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model_ranking.tflite", "wb").write(tflite_model)

interpreter = tf.lite.Interpreter(model_path="converted_model_ranking.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test the model.
if input_details[0]["name"] == "serving_default_kriteria_mentor:0":
  interpreter.set_tensor(input_details[0]["index"], np.array(["Saya mahasiswa fisika Universitas Gadjah Mada yang sangat menguasai fisika dan matematika tingkat SMA, Saya juga berpengalaman dalam menjadi asisten dosen serat asisten praktikum. Gaya mengajar saya itu mengasyikkan banget, kalian akan dibuat sangat paham untuk mengerjakan berbagai permasalahan fisika."]))
  interpreter.set_tensor(input_details[1]["index"], np.array(["Mencari tutor fisika yang berpengalaman"]))
else:
  interpreter.set_tensor(input_details[0]["index"], np.array(["Mencari tutor fisika yang berpengalaman"]))
  interpreter.set_tensor(input_details[1]["index"], np.array(["Saya mahasiswa fisika Universitas Gadjah Mada yang sangat menguasai fisika dan matematika tingkat SMA, Saya juga berpengalaman dalam menjadi asisten dosen serat asisten praktikum. Gaya mengajar saya itu mengasyikkan banget, kalian akan dibuat sangat paham untuk mengerjakan berbagai permasalahan fisika."]))

interpreter.invoke()

similarity = interpreter.get_tensor(output_details[0]['index'])
print(similarity)