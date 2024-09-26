from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from .forms import TrainingSessionForm, AnalysisForm, DoctorForm, PatientForm, HospitalForm
from sklearn.model_selection import train_test_split
from .models import TrainingSession, CustomUser,  Patient, Analysis, Hospital
import zipfile
from pathlib import Path
import os 
import json
import cv2
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam # type: ignore
from django.conf import settings 
from django.core import serializers
from django.core.paginator import Paginator
from django.shortcuts import render, get_object_or_404
from time import sleep
from django.http import JsonResponse
from tensorflow.keras.callbacks import History # type: ignore
from django.contrib.auth.decorators import user_passes_test
from django.contrib import messages


def superuser_check(user):
    return user.is_superuser


# Login view
def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            messages.success(request, 'You have successfully logged in.')
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid username or password.')
            return render(request, 'core/login.html')

    return render(request, 'core/login.html')

@login_required
def logout_view(request):
    logout(request)
    messages.success(request, 'You have successfully logged out.')
    return redirect('login')

# Dashboard view
@login_required(login_url='/login/')
def dashboard_view(request):
    analysis_list = Analysis.objects.all().order_by('-analysis_date') 

    paginator = Paginator(analysis_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number) 

    total_doctors = CustomUser.objects.filter(is_staff=True).count()  
    total_patients = Patient.objects.count() 
    total_analyses = Analysis.objects.count() 
    total_patients_analyzed = Patient.objects.filter(analyses__isnull=False).distinct().count() 

    context = {
        'current_page': 'dashboard',
        'page_obj': page_obj,  
        'total_doctors': total_doctors,
        'total_patients': total_patients,
        'total_analyses': total_analyses,
        'total_patients_analyzed': total_patients_analyzed,
    }
    
    return render(request, 'core/dashboard.html', context)

@login_required(login_url='/login/')
@user_passes_test(superuser_check)
def doctors_view(request):
    doctors_list = CustomUser.objects.all()
    paginator = Paginator(doctors_list, 10)  # Show 10 doctors per page
    page_number = request.GET.get('page')
    doctors = paginator.get_page(page_number)

    return render(request, 'core/doctors/doctors.html', {
        'current_page': 'doctors',
        'doctors': doctors
    })

@login_required(login_url='/login/')
@user_passes_test(superuser_check)
def add_doctor_view(request):
    if request.method == 'POST':
        form = DoctorForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Doctor added successfully.') 
            return redirect('doctors')
        else:
            messages.error(request, 'Please correct the errors below.') 
    else:
        form = DoctorForm()

    return render(request, 'core/doctors/add_doctor.html', {'current_page': 'doctors', 'form': form})

@login_required(login_url='/login/')
@user_passes_test(superuser_check)
def edit_doctor_view(request, doctor_id):
    doctor = get_object_or_404(CustomUser, id=doctor_id)

    if request.method == 'POST':
        form = DoctorForm(request.POST, instance=doctor)
        if form.is_valid():
            new_password = form.cleaned_data.get('password1')
            if new_password:  
                doctor.set_password(new_password)  
            form.save()
            messages.success(request, 'Doctor details updated successfully.') 
            return redirect('doctors')  
        else:
            messages.error(request, 'Please correct the errors below.') 
    else:
        form = DoctorForm(instance=doctor)

    context = {
        'form': form,
        'doctor': doctor,
    }

    return render(request, 'core/doctors/edit_doctor.html', context)

@login_required(login_url='/login/')
@user_passes_test(superuser_check)
def delete_doctor_view(request, doctor_id):
    if request.method == 'POST':
        doctor = get_object_or_404(CustomUser, id=doctor_id)
        doctor.delete()
        messages.success(request, 'Doctor deleted successfully.') 
        return redirect('doctors')  

    return redirect('doctors')


@login_required(login_url='/login/')
def patients_view(request):
     # Retrieve all patients from the database
    patients_list = Patient.objects.all().order_by('last_name')

    # Set up pagination with 10 patients per page
    paginator = Paginator(patients_list, 10)  # Show 10 patients per page

    # Get the current page number from the request GET parameters
    page_number = request.GET.get('page')
    
    # Get the patients for the current page
    patients = paginator.get_page(page_number)
    
    # Pass the paginated patients queryset to the template
    return render(request, 'core/patients/patients.html', {
        'current_page': 'patients',
        'patients': patients
    })

@login_required(login_url='/login/')
def add_patient_view(request):
    if request.method == 'POST':
        form = PatientForm(request.POST)
        if form.is_valid():
            form.save()  
            messages.success(request, 'Patient added successfully.')
            return redirect('patients') 
        else:
            messages.error(request, 'Please correct the errors below.') 
    else:
        form = PatientForm()  

    return render(request, 'core/patients/add_patient.html', {'form': form})

@login_required(login_url='/login/')
def edit_patient_view(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)

    if request.method == 'POST':
        form = PatientForm(request.POST, instance=patient)
        if form.is_valid():
            form.save() 
            messages.success(request, 'Patient details updated successfully.') 
            return redirect('patients') 
        else:
            messages.error(request, 'Please correct the errors below.') 
    else:
        form = PatientForm(instance=patient)

    context = {
        'form': form,
        'patient': patient,
    }

    return render(request, 'core/patients/edit_patient.html', context)

@login_required(login_url='/login/')
def delete_patient_view(request, patient_id):
    patient = get_object_or_404(Patient, id=patient_id)

    if request.method == 'POST':
        patient.delete() 
        messages.success(request, f'Patient {patient.first_name} {patient.last_name} was successfully deleted.') 
        return redirect('patients')  

    return redirect('patients')

@login_required(login_url='/login/')
def analysis_view(request):
    if request.method == 'POST':
        patient_code = request.POST.get('patient_code')
        image = request.FILES.get('image')
        result = "Pending"

        try:
            patient = Patient.objects.get(patient_code=patient_code)
            analysis = Analysis(
                patient=patient,
                doctor=request.user, 
                image=image,
                result=result
            )
            analysis.save()

            print(analysis)

            image_path = analysis.image.path
            print(image_path)
            tumor_types = load_labels()
            predicted_tumor_type = predict_image(image_path, tumor_types)
            analysis.result = predicted_tumor_type
            analysis.save()
            context = {
                'analysis': analysis,
                'current_page': 'analysis',
            }

            return render(request, 'core/result.html', context)

        except Patient.DoesNotExist:
            return redirect('analysis')
        
    patients = Patient.objects.all()

    patients_json = serializers.serialize('json', patients) 
    context = {
        'patients': patients_json, 
        'current_page': 'analysis',
    }
    return render(request, 'core/analysis.html', context)

@login_required(login_url='/login/')
def view_analysis_view(request, id):
    analysis = get_object_or_404(Analysis, id=id)
    context = {
        'analysis': analysis,
        'current_page': 'analysis',
    }
    
    return render(request, 'core/result.html', context)

@login_required(login_url='/login/')
def hospitals_view(request):
    hospital_list = Hospital.objects.all()
    paginator = Paginator(hospital_list, 10) 
    page_number = request.GET.get('page')
    hospitals = paginator.get_page(page_number)
    context = {
        'current_page': 'hospitals', 
        'hospitals': hospitals, 
    }
    
    return render(request, 'core/hospitals/hospitals.html', context)

@login_required(login_url='/login/')
@user_passes_test(superuser_check)
def add_hospital_view(request):
    if request.method == 'POST':
        form = HospitalForm(request.POST)
        if form.is_valid():
            form.save()
            messages.success(request, 'Hospital added successfully.') 
            return redirect('hospitals') 
        else:
            messages.error(request, 'Please correct the errors below.') 
    else:
        form = HospitalForm()

    context = {
        'form': form,
        'current_page': 'hospitals',
    }
    
    return render(request, 'core/hospitals/add_hospital.html', context)

@login_required(login_url='/login/')
@user_passes_test(superuser_check)
def edit_hospital_view(request, hospital_id):
    hospital = get_object_or_404(Hospital, id=hospital_id)

    if request.method == 'POST':
        form = HospitalForm(request.POST, instance=hospital)
        if form.is_valid():
            form.save() 
            messages.success(request, 'Hospital details updated successfully.') 
            return redirect('hospitals') 
        else:
            messages.error(request, 'Please correct the errors below.')  
    else:
        form = HospitalForm(instance=hospital)

    context = {
        'form': form,
        'hospital': hospital,
    }

    return render(request, 'core/hospitals/edit_hospital.html', context)

def delete_hospital_view(request, hospital_id):
    hospital = get_object_or_404(Hospital, id=hospital_id)

    if request.method == 'POST':
        hospital.delete() 
        messages.success(request, f'Hospital "{hospital.name}" was successfully deleted.')
        return redirect('hospitals') 
    return redirect('hospitals')

@login_required(login_url='/login/')
@user_passes_test(superuser_check)
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

            model = create_or_load_model(model_exists, learning_rate, len(dataset_labels))
        
            history = train_model(model, training_images, training_labels, epochs)

            # Save the trained model
            save_model(model)

            # Convert the training history to a serializable format (JSON)
            history_dict = {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss'],
                'accuracy': history.history['accuracy'],
                'val_accuracy': history.history['val_accuracy'],
            }
            messages.success(request, 'Training completed successfully.')
            # Return the history data as JSON
            return JsonResponse({'success': True, 'history': history_dict})
    else:
        form = TrainingSessionForm()

    return render(request, 'core/training.html', {'form': form, 'current_page': 'training'})

@login_required(login_url='/login/')
@user_passes_test(superuser_check)
def training_results_view(request):
    history = request.GET.get('history', None)
    
    if history:
        history = json.loads(history)  # Convert JSON string to a Python dictionary

    return render(request, 'core/training_results.html', {'history': history, 'current_page': 'training'})


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
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
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
    history = model.fit(
        training_images, 
        training_labels, 
        epochs=epochs,
        validation_split=0.1, 
        verbose=1  # Set to 1 for detailed logging
    )
    return history

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

        return predicted_tumor_type