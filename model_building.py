# model_building.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from data_preprocessing import create_generators

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, train_generator, validation_generator):
    history = model.fit(
        train_generator,
        epochs=10,
        validation_data=validation_generator
    )
    return history

def save_model(model, filename='cleanliness_model.h5'):
    model.save(filename)

if __name__ == "__main__":
    dataset_path = 'dataset'
    train_gen, val_gen = create_generators(dataset_path)
    model = build_model()
    history = train_model(model, train_gen, val_gen)
    save_model(model)

# # Tambahkan di model_building.py setelah pelatihan model
# def evaluate_model(model, validation_generator):
#     evaluation = model.evaluate(validation_generator)
#     print(f'Validation Loss: {evaluation[0]}')
#     print(f'Validation Accuracy: {evaluation[1]}')

# if __name__ == "__main__":
#     dataset_path = 'dataset'
#     train_gen, val_gen = create_generators(dataset_path)
#     model = build_model()
#     history = train_model(model, train_gen, val_gen)
#     save_model(model)
#     evaluate_model(model, val_gen)  # Evaluasi model

# # Tambahkan di model_building.py setelah evaluasi model
# import matplotlib.pyplot as plt

# def plot_history(history):
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     plt.show()

#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

# if __name__ == "__main__":
#     dataset_path = 'dataset'
#     train_gen, val_gen = create_generators(dataset_path)
#     model = build_model()
#     history = train_model(model, train_gen, val_gen)
#     save_model(model)
#     evaluate_model(model, val_gen)  # Evaluasi model
#     plot_history(history)  # Visualisasi hasil
