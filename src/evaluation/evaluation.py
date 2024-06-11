import requests
import json

# URL API TensorFlow Serving
url = 'http://localhost:8601/v1/models/saved_model:predict'

# Data yang akan dikirimkan ke model
data = {
    "instances": ["Saya ingin belajar fisika dengan mahasiswa fisika yang sudah berpengalaman menjadi asisten dosen dan juga di daerah Yogyakarta dengan menggunakan bahasa indonesia harus laki-laki."]
}

# Mengirimkan permintaan POST
response = requests.post(url, json=data)
predictions = response.json()

# Memeriksa dan menampilkan hasil prediksi
recomendation = predictions['predictions'][0]['output_2'][:3]
print(recomendation)