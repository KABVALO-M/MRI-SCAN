import os
import zipfile
import json
from django.shortcuts import render
from django.http import HttpResponse
from .forms import TrainingForm
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

LABELS_FILE = 'labels.json'
MODEL_FILE = 'trained_model.h5'

# Helper function to load or initialize labels
def load_labels():
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            labels = json.load(f)
    else:
        labels = {}  # Empty dictionary to store labels
    return labels

def save_labels(labels):
    with open(LABELS_FILE, 'w') as f:
        json.dump(labels, f)

def unzip_file(file, extract_to='training_data'):
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def get_model(input_shape=(150, 150, 3), num_classes=2):
    # Load existing model or create a new one if it doesn't exist
    if os.path.exists(MODEL_FILE):
        model = load_model(MODEL_FILE)
    else:
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(request):
    if request.method == 'POST':
        form = TrainingForm(request.POST, request.FILES)
        if form.is_valid():
            # Get form data
            label_name = form.cleaned_data['label_name']
            epochs = form.cleaned_data['epochs']
            validation_split = form.cleaned_data['validation_split']
            train_file = request.FILES['train_file']
            
            # Unzip the training data
            unzip_file(train_file)

            # Load or initialize labels
            labels = load_labels()

            # Assign new index if label is new
            if label_name not in labels:
                labels[label_name] = len(labels)
                save_labels(labels)

            # Update the number of classes in the model
            num_classes = len(labels)

            # Load or initialize the model
            model = get_model(num_classes=num_classes)

            # Create ImageDataGenerator for training and validation
            datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)
            train_gen = datagen.flow_from_directory('training_data',
                                                    target_size=(150, 150),
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    subset='training')
            val_gen = datagen.flow_from_directory('training_data',
                                                  target_size=(150, 150),
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  subset='validation')

            # Train the model incrementally
            model.fit(train_gen, epochs=epochs, validation_data=val_gen)

            # Save the updated model
            model.save(MODEL_FILE)

            return HttpResponse(f'Model updated and trained with label: {label_name}')
    else:
        form = TrainingForm()

    return render(request, 'train.html', {'form': form})






So now that we are saving the files in zip format, we should begin the training process implementation step by step, we will start by unziping the uploaded folder, and get the directory name and path which contains the images, if there are many directories, get the names of those directories and keep them as labels, meaning to say if there are multiple levels of directories, get their paths and names and keep them so we can join them when we want to get the images, make sure you get the directory name which has the actual images, I hope it is well understood


The uploaded folder (ZIP file) will be unzipped.
The directory structure of the unzipped content will be analyzed.
The paths and names of the directories will be recorded, especially if they contain image files.
If there are multiple levels of directories, their names should be stored as labels.
Only directories that contain actual image files will be processed.