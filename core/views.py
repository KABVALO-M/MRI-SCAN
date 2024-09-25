from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import TrainingSessionForm, AnalysisForm, DoctorForm
from sklearn.model_selection import train_test_split
from .models import TrainingSession, CustomUser
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
    doctors = CustomUser.objects.all()
    return render(request, 'core/doctors/doctors.html', {
        'current_page': 'doctors',
        'doctors': doctors
    })

def add_doctor_view(request):
    if request.method == 'POST':
        form = DoctorForm(request.POST)
        if form.is_valid():
            form.save()  # Save the new doctor record to the database
            return redirect('doctors')  # Redirect to the doctor list page after submission
    else:
        form = DoctorForm()

    return render(request, 'core/doctors/add_doctor.html', {'current_page': 'doctors','form': form})

@login_required(login_url='/login/')
def patients_view(request):
    return render(request, 'core/patients.html', {'current_page': 'patients'})

@login_required(login_url='/login/')
def analysis_view(request):
    # if request.method == 'POST':
    #     form = AnalysisForm(request.POST, request.FILES)
    #     if form.is_valid():
    #         # Get the uploaded image file and confidence threshold
    #         image_file = form.cleaned_data['image_file']
    #         confidence_threshold = form.cleaned_data['confidence_threshold']

    #         # Save the uploaded image to the database
    #         image_upload = ImageUpload(image=image_file)
    #         image_upload.save()  # Save the image instance to the database

    #         # Get the path to the uploaded image
    #         image_path = image_upload.image.path  # This gives the full path to the uploaded image file
    #         tumor_types = load_labels()
    #         # Call your prediction function
    #         predicted_index, predicted_tumor_type = predict_image(image_path, tumor_types)

    #         return render(request, 'core/result.html', {
    #             'prediction_index': predicted_index,
    #             'prediction_tumor_type': predicted_tumor_type
    #         })
    #     else:
    #         # Handle form errors
    #         return render(request, 'core/analysis.html', {'form': form})

    # # If GET request, display the empty form
    # form = AnalysisForm()
    return render(request, 'core/analysis.html')

@login_required(login_url='/login/')
def training_view(request):
    model_path = get_model_path()
    model_exists = os.path.exists(model_path)

    if request.method == 'POST':
        form = TrainingSessionForm(request.POST, request.FILES)
        
        if form.is_valid():
            training_session = form.save(commit=False)
            training_file = form.cleaned_data['training_file']
            epochs = form.cleaned_data['epochs']
            batch_size = form.cleaned_data['batch_size']
            learning_rate = float(form.cleaned_data['learning_rate'])
            label_info = form.cleaned_data['label_info']
            
            training_session.user = request.user
            training_session.save()

            extracted_directory, file_name_without_extension = handle_uploaded_file(training_file)
            existing_labels, dataset_labels = determine_labels(extracted_directory, label_info)
            print(f"Determined labels: {dataset_labels}")

            image_data, label_data = load_and_preprocess_images(extracted_directory, dataset_labels, file_name_without_extension)
            shuffled_images, shuffled_labels = shuffle(image_data, label_data, random_state=101)

            # Split the dataset
            training_images, testing_images, training_labels, testing_labels = split_data(shuffled_images, shuffled_labels)
            print(f"Training images shape: {training_images.shape}, Training labels shape: {training_labels.shape}")

            # Convert labels to categorical
            training_labels = convert_labels_to_categorical(training_labels, existing_labels)
            testing_labels = convert_labels_to_categorical(testing_labels, existing_labels)

            # Load or create the model
            print(len(dataset_labels))
            model = create_or_load_model(model_exists, learning_rate, len(dataset_labels))

            # Train the model
            train_model(model, training_images, training_labels, epochs)

            # Save the trained model
            save_model(model)

            # Save Labels
            # save_labels(dataset_labels)

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

# Define the path where the model will be saved (inside the project base dir)
def get_model_path():
    model_dir = os.path.join(settings.BASE_DIR, 'model')
    model_path = os.path.join(model_dir, 'trained_model.keras')
    
    # Check if the model directory exists, and if not, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    return model_path

def handle_uploaded_file(training_file):
    """Unzip the uploaded zip file and return the extraction path."""
    if zipfile.is_zipfile(training_file):
        extract_base_path = unzip_file(training_file)
        zip_filename = os.path.basename(training_file.name)
        file_name_without_extension = os.path.splitext(zip_filename)[0]
        extracted_directory = os.path.join(extract_base_path, file_name_without_extension)
        print(f"Extracted files to: {extracted_directory}")
        return extracted_directory, file_name_without_extension
    return None, None

def determine_labels(extracted_directory, label_info):
    """Determine labels based on subdirectories and provided label info."""
    subdirectories = [d for d in os.listdir(extracted_directory) if os.path.isdir(os.path.join(extracted_directory, d))]
    existing_labels = load_labels()
    print("Existing Labels: ",existing_labels)
    # If no subdirectories and no label_info, return the directory name as a single label
    if not subdirectories and not label_info:
        dataset_labels = [os.path.basename(extracted_directory)]
    else:
        dataset_labels = [label_info] if label_info else subdirectories
    
    # Check if each dataset label exists in existing labels, if not, append it
    for label in dataset_labels:
        if label not in existing_labels:
            existing_labels.append(label)
    
    return existing_labels, dataset_labels

def load_and_preprocess_images(extracted_directory, labels, file_name_without_extension):
    """Load and preprocess images from the extracted directory based on labels."""
    image_data = []
    label_data = []
    image_size = 150

    # Check if there are subdirectories
    subdirectories = [d for d in os.listdir(extracted_directory) if os.path.isdir(os.path.join(extracted_directory, d))]

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

    return np.array(image_data), np.array(label_data)

def split_data(shuffled_images, shuffled_labels):
    """Perform train-test split on the dataset."""
    return train_test_split(shuffled_images, shuffled_labels, test_size=0.1, random_state=101)

def convert_labels_to_categorical(labels, combined_labels):
    """Convert label arrays to categorical format."""
    label_indices = [combined_labels.index(label) for label in labels]
    print(label_indices)
    return tf.keras.utils.to_categorical(label_indices)

def create_or_load_model(model_exists, learning_rate, num_classes):
    """Create a new model or load an existing one."""
    from tensorflow.keras.models import load_model, Sequential # type: ignore
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
    
    model_path = get_model_path()
    if model_exists:
        model = load_model(model_path)
         # Set appropriate loss function
        if num_classes == 2:
            loss = 'binary_crossentropy'
        else:
            loss = 'categorical_crossentropy'
        
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
    else:
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


        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Summary of the model
        model.summary()

        print("Starting new model training...")
    
    return model

def train_model(model, training_images, training_labels, epochs):
    """Train the model with the training dataset."""
    return model.fit(training_images, training_labels, epochs=epochs, validation_split=0.1)

def save_model(model):
    """Save the trained model."""
    model_path = get_model_path()
    model.save(model_path)
    print("Model training completed and saved.")

# def save_labels(labels, filename='labels.txt'):
#     """Save the labels to a file."""
#     with open(filename, 'a') as file:  # Open in append mode
#         for label in labels:
#             file.write(f"{label}\n")

def load_labels(filename='labels.txt'):
    """Load labels from a file."""
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return [line.strip() for line in file.readlines()]
    return []


def predict_image(image_path, tumor_types):
    model_path = get_model_path()
    model_exists = os.path.exists(model_path)
    if model_exists:
        from tensorflow.keras.models import load_model # type: ignore
        import cv2
        import numpy as np

        model = load_model(model_path)

        # Read the image from the file
        img = cv2.imread(image_path)

        # Resize the image to the input size required by your model (example: 150x150)
        img_resized = cv2.resize(img, (150, 150))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction using the model
        prediction = model.predict(img_array)

        # Assuming binary classification or multi-class classification:
        predicted_index = np.argmax(prediction, axis=1)[0]  # Get the index of the predicted class

        # Return the predicted class index and its corresponding tumor type
        predicted_tumor_type = tumor_types[predicted_index]

        return predicted_index, predicted_tumor_type