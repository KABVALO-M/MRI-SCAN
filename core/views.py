from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import TrainingSessionForm
from sklearn.model_selection import train_test_split
from .models import TrainingSession
import zipfile
from pathlib import Path
import os 
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam # type: ignore
from django.conf import settings 

# Login view
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        # Authenticate the user
        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Log the user in
            login(request, user)
            # Redirect to the home page or dashboard after successful login
            return redirect('dashboard')
        else:
            # Invalid credentials, show an error
            error = "Invalid username or password."
            return render(request, 'core/login.html', {'error': error})

    return render(request, 'core/login.html')

@login_required
def logout_view(request):
    logout(request)
    return redirect('login')

# Dashboard view
@login_required(login_url='/login/')
def dashboard_view(request):
    return render(request, 'core/dashboard.html', {'current_page': 'dashboard'})

@login_required(login_url='/login/')
def doctors_view(request):
    return render(request, 'core/doctors.html', {'current_page': 'doctors'})

@login_required(login_url='/login/')
def patients_view(request):
    return render(request, 'core/patients.html', {'current_page': 'patients'})

@login_required(login_url='/login/')
def analysis_view(request):
    return render(request, 'core/analysis.html', {'current_page': 'analysis'})

@login_required(login_url='/login/')
def training_view(request):
    # Define the path where the model will be saved (inside the project base dir)
    model_dir = os.path.join(settings.BASE_DIR, 'model')
    model_path = os.path.join(model_dir, 'trained_model.h5')  


    # Check if the model directory exists, and if not, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Check if the model already exists in the specified path
    model_exists = os.path.exists(model_path)


    if request.method == 'POST':
        # Use the updated TrainingSessionForm
        form = TrainingSessionForm(request.POST, request.FILES)
        
        if form.is_valid():
             # Save the form data into the model
            training_session = form.save(commit=False)

            # Extract the cleaned data
            training_file = form.cleaned_data['training_file']
            epochs = form.cleaned_data['epochs']
            batch_size = form.cleaned_data['batch_size']
            learning_rate = float(form.cleaned_data['learning_rate'])
            label_info = form.cleaned_data['label_info']
            print(learning_rate)
            

            # Assign the current user to the training session (assuming a foreign key relationship)
            training_session.user = request.user

            # Save the model to the database
            training_session.save()
# ===============================================================================================================================================================================
            # Step 1: Unzip the uploaded zip file
            if zipfile.is_zipfile(training_file):
                # Unzip the file to the base extraction directory
                extract_base_path = unzip_file(training_file)

                # Get the name of the uploaded zip file without the extension
                zip_filename = os.path.basename(training_file.name)
                file_name_without_extension = os.path.splitext(zip_filename)[0]

                # Construct the full extraction path
                extracted_directory = os.path.join(extract_base_path, file_name_without_extension)
                print(f"Extracted files to: {extracted_directory}")
# ==============================================================================================================================================================================
                # Step 2: Check for subdirectories and set the label
                # List the directories in the extracted directory
                subdirectories = [d for d in os.listdir(extracted_directory) if os.path.isdir(os.path.join(extracted_directory, d))]

                # Determine the labels based on the presence of subdirectories and label_info
                if not subdirectories and not label_info:
                    # Use file_name_without_extension as the label if there are no subdirectories and label_info is not provided
                    labels = [file_name_without_extension]  # Store as a list
                else:
                    # Use the provided label_info if it exists or the list of subdirectory names if available
                    labels = [label_info] if label_info else subdirectories
                
                num_classes = len(labels)

                # Add logic here to save or use the labels as needed
                print(f"Determined labels: {labels}")
# =============================================================================================================================================================================
                # Step 3: Gather images based on the determined labels
                image_data = []  # List to hold image data
                label_data = []  # List to hold corresponding labels
                image_size = 150

                # Check if there are subdirectories or not
                if subdirectories:
                    # Case when there are subdirectories
                    for label in labels:
                        folder_path = os.path.join(extracted_directory, label)
                        if os.path.exists(folder_path):
                            for img_file in os.listdir(folder_path):
                                print(img_file)  # For debugging: prints the image file name
                                img_path = os.path.join(folder_path, img_file)
                                img = cv2.imread(img_path)
                                if img is not None:
                                    img = cv2.resize(img, (image_size, image_size))
                                    image_data.append(img)
                                    label_data.append(label)
                        else:
                            print(f"Warning: Directory {folder_path} does not exist.")
                else:
                    # Case when there are no subdirectories
                    # Collect images directly from the extracted directory
                    for img_file in os.listdir(extracted_directory):
                        print(img_file)  # For debugging: prints the image file name
                        img_path = os.path.join(extracted_directory, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            img = cv2.resize(img, (image_size, image_size))
                            image_data.append(img)
                            # Use the file name or the parent directory as a single label
                            label_data.append(file_name_without_extension)

                # Convert to numpy arrays
                image_array = np.array(image_data)        # Renamed from train_images to image_array
                label_array = np.array(label_data)        # Renamed from train_labels to label_array

# ==============================================================================================================================================================================
                # Step 4: Shuffle the dataset
                shuffled_images, shuffled_labels = shuffle(image_array, label_array, random_state=101)

                # Check the shape of the training data
                print(f"Shuffled train_images shape: {shuffled_images.shape}")
# ==============================================================================================================================================================================
                # Step 5: Perform train-test split
                training_images, testing_images, training_labels, testing_labels = train_test_split(
                    shuffled_images, 
                    shuffled_labels, 
                    test_size=0.1, 
                    random_state=101
                )

                # Check the shape of the split data
                print(f"Training images shape: {training_images.shape}, Training labels shape: {training_labels.shape}")
                print(f"Testing images shape: {testing_images.shape}, Testing labels shape: {testing_labels.shape}")
# ==============================================================================================================================================================================
                # Convert labels to categorical format
                # Create new label arrays to convert labels to integers
                training_labels_new = []
                for label in training_labels:
                    training_labels_new.append(labels.index(label))  # Find index of label in the labels list
                training_labels = training_labels_new
                training_labels = tf.keras.utils.to_categorical(training_labels)  # Convert to categorical

                testing_labels_new = []
                for label in testing_labels:
                    testing_labels_new.append(labels.index(label))  # Find index of label in the labels list
                testing_labels = testing_labels_new
                testing_labels = tf.keras.utils.to_categorical(testing_labels)  # Convert to categorical

                # Check the shapes of the final data
                print(f"Final training images shape: {training_images.shape}, Training labels shape: {training_labels.shape}")
                print(f"Final testing images shape: {testing_images.shape}, Testing labels shape: {testing_labels.shape}")

# ==============================================================================================================================================================================
            if model_exists:
                print("Continuing training on the existing model...")
                # Load and recompile the model with a new optimizer
                from tensorflow.keras.models import load_model # type: ignore
                print("Continuing training on the existing model...")
                model = load_model(model_path)

                # Recompile the model with the same loss function and a new optimizer instance
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
            else:
                from tensorflow.keras.models import Sequential # type: ignore
                from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
                # Define the model
                model = Sequential()

                # Adding convolutional layers
                model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))  # Input shape for RGB images
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(MaxPooling2D(2, 2))
                model.add(Dropout(0.3))

                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(Conv2D(64, (3, 3), activation='relu'))
                model.add(Dropout(0.3))
                model.add(MaxPooling2D(2, 2))

                model.add(Dropout(0.3))
                model.add(Conv2D(128, (3, 3), activation='relu'))
                model.add(Conv2D(128, (3, 3), activation='relu'))
                model.add(Conv2D(128, (3, 3), activation='relu'))
                model.add(MaxPooling2D(2, 2))
                model.add(Dropout(0.3))

                model.add(Conv2D(128, (3, 3), activation='relu'))
                model.add(Conv2D(256, (3, 3), activation='relu'))
                model.add(MaxPooling2D(2, 2))
                model.add(Dropout(0.3))

                # Flattening the output before feeding into Dense layers
                model.add(Flatten())
                model.add(Dense(512, activation='relu'))
                model.add(Dense(512, activation='relu'))
                model.add(Dropout(0.3))

                # Output layer for classification
                model.add(Dense(num_classes, activation='softmax'))  # 4 output classes

                # # Compile the model
                model.compile(optimizer='adam', 
                            loss='categorical_crossentropy', 
                            metrics=['accuracy'])

                # Summary of the model
                model.summary()

                print("Starting new model training...")
               
            # Step 7: Train the model
            history = model.fit(
                training_images, training_labels, 
                epochs=epochs, 
                validation_split=0.1
            )

            # Save the model after training
            model.save(model_path)

            print("Model training completed and saved.")
            # Redirect to the dashboard or relevant page after processing
            return redirect('dashboard')

    else:
        form = TrainingSessionForm()

    return render(request, 'core/training.html', {'form': form, 'current_page': 'training'})




@login_required(login_url='/login/')
def settings_view(request):
    return render(request, 'core/settings.html', {'current_page': 'settings'})

@login_required(login_url='/login/')
def profile_view(request):
    return render(request, 'core/profile.html', {'current_page': 'profile'})







def unzip_file(zip_file, extract_to='uploads/extracted/'):
    """
    Unzips the given zip file into the specified directory.
    
    :param zip_file: The uploaded zip file.
    :param extract_to: The directory where the zip content will be extracted.
    :return: Path to the extraction directory.
    """
    # Ensure the extraction directory exists
    Path(extract_to).mkdir(parents=True, exist_ok=True)
    
    # Unzip the file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    return extract_to  # Return the base extraction directory


def get_image_directories(base_dir='uploads/extracted/'):
    labels = {}  # Dictionary to hold labels and their respective paths
    # Traverse through the base directory
    for root, dirs, files in os.walk(base_dir):
        # Filter out files and check for images
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        if image_files:  # If there are image files in this directory
            # Get the directory name relative to base_dir
            relative_dir = os.path.relpath(root, base_dir)
            labels[relative_dir] = root  # Store the path associated with its label

    return labels