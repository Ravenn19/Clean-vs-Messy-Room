import cv2
import numpy as np
from tensorflow.keras.models import load_model

def load_and_process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image at path {image_path} could not be loaded. Check the file path and try again.")
    img = cv2.resize(img, (150, 150))
    img = img.astype('float32') / 255
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(model, image_path):
    img = load_and_process_image(image_path)
    prediction = model.predict(img)
    return 'Clean' if prediction[0] < 0.5 else 'Dirty'

if __name__ == "__main__":
    model = load_model('cleanliness_model.h5')
    image_path = 'dataset\dirty\image4.jpg'  # Ganti dengan path gambar yang ingin diuji
    result = predict_image(model, image_path)
    print(f'The image is {result}')
