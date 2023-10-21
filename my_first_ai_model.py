import tensorflow as tf

# Görüntü yükle
image = cv2.imread("image.jpg")

# Nesneleri tanımak için bir model eğit
model = tf.keras.models.load_model("model.h5")

# Nesneleri görüntü üzerinde tespit
predictions = model.predict(image)

# Nesneleri görüntü üzerinde çiz
for prediction in predictions:
    (x, y, w, h, class_id, confidence) = prediction
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.putText(image, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Görüntüyü göster
cv2.imshow("Image", image)
cv2.waitKey(0)
