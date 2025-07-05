# data_preprocessing.py
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(dataset_path):
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalisasi pixel gambar
        validation_split=0.2  # Split data menjadi training dan validation
    )

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',   
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

if __name__ == "__main__":
    dataset_path = 'dataset'
    train_gen, val_gen = create_generators(dataset_path)
