import tensorflow as tf
import numpy as np
model_path = './save_models/ranking_models/1'

loaded = tf.saved_model.load(model_path)

test_similarity = {}
test_kriteria_mentor = ["Saya mahasiswa fisika Universitas Gadjah Mada yang sangat menguasai fisika dan matematika tingkat SMA, Saya juga berpengalaman dalam menjadi asisten dosen serat asisten praktikum. Gaya mengajar saya itu mengasyikkan banget, kalian akan dibuat sangat paham untuk mengerjakan berbagai permasalahan fisika.",
                        "Saya berpengalaman dalam mengajar fisika dan matematika smp serta sudah memiliki sertifikat resmi",
                        "Saya menggunakan beberapa warna dalam mencatat materi tutor agar lebih menarik dan tidak bosan dilihat."]
for kriteria_mentor in test_kriteria_mentor:
  test_similarity[kriteria_mentor] = loaded({
        "kriteria_mentor_user": tf.constant(["Mencari tutor fisika yang berpengalaman"], dtype=tf.string),
        "kriteria_mentor": tf.constant([kriteria_mentor], dtype=tf.string),
  })

print("Similarity:")
for kriteria_mentor, score in sorted(test_similarity.items(), key=lambda x: x[1], reverse=True):
  print(f"{kriteria_mentor}: {score}")