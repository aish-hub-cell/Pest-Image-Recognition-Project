import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import os
import h5py
import gzip
import shutil

dataset_path = r"C:\Users\Ayesha\Downloads\archive (2)\farm_insects"


if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset folder not found at {dataset_path}")


datagen = ImageDataGenerator(
    rescale=1.0/255, 
    validation_split=0.2,  
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="training"
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    subset="validation"
)


model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5), 
    Dense(len(train_generator.class_indices), activation="softmax")  
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=15  
)


model.save("pest_disease_recognition.h5")
print("Model training complete and saved as 'pest_disease_recognition.h5'")

with open("pest_disease_recognition.h5", "rb") as f_in:
    with gzip.open("pest_disease_recognition.h5.gz", "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

print("Model compressed successfully.")